import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''Gumbel-Max sampling with float64 precision.'''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''Calculate tokens to transfer at each step for uniform denoising.'''
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, constraints=None):
    '''Standard LLaDA generation without early exit.'''
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # Apply constraints
    if constraints is not None:
        for pos, token_id in constraints.items():
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[:, absolute_pos] = token_id
    
    prompt_index = (x != mask_id)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            # Forward pass
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
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Compute confidence
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # Mask out tokens beyond current block
            x0_p[:, block_end:] = -np.inf
            
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Transfer tokens
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            
            # Maintain constraints
            if constraints is not None:
                for pos, token_id in constraints.items():
                    absolute_pos = prompt.shape[1] + pos
                    if absolute_pos < x.shape[1]:
                        x[:, absolute_pos] = token_id
    
    return x


def main():
    device = 'cuda'
    
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    
    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()