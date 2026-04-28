import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, constraints=None,
             track_positions=None, return_history=False,
             early_exit_positions=None, early_exit_window=5, early_exit_variation=0.05):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        constraints: Dictionary mapping positions to token IDs for forced generation.
    '''
    # -------- optional confidence tracking --------
    if track_positions is None:
        track_positions = []
    if return_history:
        # dictionary mapping position -> list[float] that stores the confidence
        # (softmax probability of the currently selected token) at every sampling step
        conf_history = {pos: [] for pos in track_positions}
    # ----------------------------------------------
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Apply constraints to the initial state
    if constraints is not None:
        for pos, token_id in constraints.items():
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[:, absolute_pos] = token_id

    prompt_index = (x != mask_id)

    print('gen_length', gen_length, 'block_length', block_length)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # early-exit tracking
    if early_exit_positions is None:
        early_exit_positions = []
    early_exit_enabled = len(early_exit_positions) > 0
    early_exit_step_global = None

    x0_history = []  # List of lists: x0_history[block][step] = x0_cpu
    true_indices_history = []  # List of lists: true_indices_history[block][step] = true_indices_cpu

    for num_block in range(num_blocks):
        early_exit = False
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        block_x0_history = []
        block_true_indices_history = []
    
        for i in range(steps):
            mask_index = (x == mask_id)
            mask_index_current = mask_index
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            x0_cpu = x0.detach().cpu()
            block_x0_history.append(x0_cpu)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            
            # store confidence for the requested positions
            if return_history:
                for pos in track_positions:
                    # safeguard against out‑of‑range positions
                    if pos < x0_p.shape[1]:
                        # batch size is 1 in typical use; take the first element
                        conf_history[pos].append(float(x0_p[0, pos].item()))
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            true_indices = torch.nonzero(transfer_index, as_tuple=False)
            true_indices_cpu = true_indices.detach().cpu()
            block_true_indices_history.append(true_indices_cpu)
            
            x[transfer_index] = x0[transfer_index]

            # Ensure constraints are maintained after each step
            if constraints is not None:
                for pos, token_id in constraints.items():
                    absolute_pos = prompt.shape[1] + pos
                    if absolute_pos < x.shape[1]:
                        x[:, absolute_pos] = token_id

            # ---------------- Early-exit check ----------------
            if early_exit_enabled and return_history:
                # ensure we have enough history
                if len(next(iter(conf_history.values()))) >= early_exit_window:
                    stable = True
                    for pos in early_exit_positions:
                        if pos not in conf_history:
                            continue
                        window_slice = conf_history[pos][-early_exit_window:]
                        if max(window_slice) - min(window_slice) > early_exit_variation:
                            stable = False
                            break
                    if stable:
                        early_exit_step_global = num_block * steps + i
                        # debug print for early-exit – shows position confidences in the window
                        if return_history:
                            print(f"[Generate] early-exit at global step {early_exit_step_global} (window={early_exit_window}, var={early_exit_variation})")
                            for pos in early_exit_positions:
                                if pos in conf_history:
                                    win_slice = conf_history[pos][-early_exit_window:]
                                    win_str = ", ".join(f"{v:.3f}" for v in win_slice)
                                    print(f"  pos {pos}: [{win_str}]")
                        # finalize: fill all still-masked positions with current best predictions
                        x[mask_index_current] = x0[mask_index_current]
                        early_exit = True
                else:
                    early_exit = False
            else:
                early_exit = False
            # ---------------------------------------------------
            if early_exit:
                break
            
        x0_history.append(block_x0_history)
        true_indices_history.append(block_true_indices_history)
                
        if early_exit:
            break

    if return_history:
        if early_exit_step_global is not None:
            conf_history['early_step'] = early_exit_step_global
        for i in range(len(x0_history)):
            x0_history[i] = torch.cat(x0_history[i], dim=0)
        
        return x, conf_history, x0_history, true_indices_history
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
