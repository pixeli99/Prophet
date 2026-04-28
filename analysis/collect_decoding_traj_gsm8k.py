import re
from typing import List, Tuple, Dict
import os
import numpy as np
import torch
import tqdm
import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from generate import generate

def parse_args():
    parser = argparse.ArgumentParser(description='Decoding configuration')
    
    parser.add_argument('--decode_policy', 
                       type=str, 
                       default='random',
                       choices=['random', 'low_confidence'],
                       help='Decoding policy to use')
    
    parser.add_argument('--constraint_policy', 
                       type=str, 
                       default='constraint',
                       choices=['constraint', 'none'],
                       help='Constraint policy to apply')
    
    parser.add_argument('--blocklen', 
                       type=int, 
                       default=8,
                       help='Block length for decoding')
    
    parser.add_argument('--gen_length', 
                       type=int, 
                       default=256,
                       help='Generation length (total timesteps)')
    
    parser.add_argument('--range_lst', 
                       type=int,
                       nargs='+', default=[0, 1500])
    
    return parser.parse_args()

args = parse_args()

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


# --- add word-constraint setup ----------------
CONSTRAINTS_TEXT = "220:Answer" 

def _parse_constraints(text: str, tokenizer) -> Dict[int, int]:
    constraints = {}
    for part in text.split(','):
        if ':' not in part:
            continue
        pos_str, word = part.split(':', 1)
        pos = int(pos_str.strip())
        word = word.strip()
        # tokenizer expects leading space for standalone words as in vis.py
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, tid in enumerate(ids):
            constraints[pos + i] = tid
    return constraints

# will be initialised later once tokenizer is available
CONSTRAINTS = {}

def extract_answer_from_solution(solution: str) -> str:
    """GSM8K answers are in the form of a free-form rationale ending with "#### <answer>"."""
    m = re.search(r"####\s*(-?\d+[\d,]*)", solution)
    if m:
        return m.group(1)
    # Fallback – sometimes the answer is on a new line without the hashes.
    # This is a best-effort heuristic.
    numbers = re.findall(r"-?\d+[\d,]*", solution)
    return numbers[-1] if numbers else ""


def extract_answer_from_prediction(pred: str) -> str:
    """Extract the answer string after "Answer:" (case-insensitive); fallback to last number."""
    ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
    m = re.search(ANSWER_PATTERN, pred)
    if m:
        return m.group(1).replace("$", "")
    numbers = re.findall(r"-?\d+[\d,]*", pred)
    return numbers[-1].replace("$", "") if numbers else ""


def save_question_histories(save_dir, 
                            question_idx, 
                            steps, 
                            x0_history, 
                            true_indices_history,
                            correct,
                            pred_ans,
                            pred_text,
                            ans_posidx,
                            gt_text,
                            pred_token_id,
                            gt_token_id,
                            prompt_token_len,
                            gen_ids):
    """Save only x0_history and true_indices_history for a specific question to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    data = {
        'x0_history': x0_history,
        'true_indices_history': true_indices_history,
        'correct': correct,
        'pred_text': pred_text,
        'ans_posidx': ans_posidx,
        'pred_ans': pred_ans,
        'gt_text': gt_text,
        'pred_token_id': pred_token_id,
        'gt_token_id': gt_token_id,
        'prompt_token_len': prompt_token_len,
        'gen_ids': gen_ids
    }
    
    filename = f"question_{question_idx:04d}_steps_{steps:03d}.pt"
    filepath = os.path.join(save_dir, filename)
    
    torch.save(data, filepath)
    
    print(f"Saved question {question_idx} histories to {filepath}")


def find_subsequence_index(haystack, needle):
    """Find the last starting index of needle subsequence in haystack list."""
    if not needle:
        return -1
    
    last_index = -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            last_index = i
    
    return last_index

def evaluate_on_steps(
    model,
    tokenizer,
    questions: List[str],
    ground_truths: List[str],
    steps_list: List[int],
    gen_length: int = 128,
    device: str = "cuda",
    block_length: int = 128,
) -> Tuple[List[float], List[float]]:
    """Returns accuracy list and confidence list (mean over dataset for each step)."""
    accuracy_per_step = []
    confidence_per_step = []

    track_positions = list(range(gen_length))  # track confidence for every generated position

    for steps in steps_list:
        correct = 0
        confs: List[float] = []
        pbar = tqdm.tqdm(total=args.range_lst[-1]-args.range_lst[0])
        for q_idx, (q, gt) in enumerate(zip(questions, ground_truths)):
            # Build prompt – we directly feed the raw question.
            if q_idx not in list(range(args.range_lst[0], args.range_lst[-1])):
                continue
            
            input_ids = tokenizer(q)["input_ids"]
            prompt = torch.tensor(input_ids, device=device).unsqueeze(0)

            if args.constraint_policy == 'constraint':
                print('using constraints')
                const_input = CONSTRAINTS
            else:
                print('w/o constraints')
                const_input = None
                
            out, conf_hist, x0_history, true_indices_history = generate(
                model,
                prompt,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking=args.decode_policy,
                constraints=const_input,
                track_positions=track_positions,
                return_history=True,
            )

            gen_ids = out[:, prompt.shape[1] :][0].tolist()
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            pred_ans = extract_answer_from_prediction(pred_text)

            pred_ans_id = tokenizer.encode(pred_ans, add_special_tokens=False)
            
            pred_ans_index = prompt.shape[1] + find_subsequence_index(gen_ids, pred_ans_id)
            
            if pred_ans == gt:
                correct += 1
            
            gt_token_id = tokenizer.encode(gt, add_special_tokens=False)
  
  
            save_question_histories(
                save_dir=f"./question_histories_{args.decode_policy}_{args.constraint_policy}_index_genlen_step{args.gen_length}_blocklen{args.blocklen}",
                question_idx=q_idx,
                steps=steps,
                x0_history=x0_history,
                true_indices_history=true_indices_history,
                correct= pred_ans == gt,
                pred_text = pred_text,
                pred_ans = pred_ans,
                gt_text = gt, 
                pred_token_id = pred_ans_id,
                gt_token_id = gt_token_id,
                ans_posidx=pred_ans_index,
                prompt_token_len=prompt.shape[1],
                gen_ids=gen_ids
            )

            # locate positions of answer tokens (best effort)
            if pred_ans:
                ans_token_ids = tokenizer(pred_ans, add_special_tokens=False)["input_ids"]
                # naive subsequence search
                rel_pos = -1
                for i in range(len(gen_ids) - len(ans_token_ids) + 1):
                    if gen_ids[i : i + len(ans_token_ids)] == ans_token_ids:
                        rel_pos = i
                        break
                if rel_pos != -1:
                    # take confidence at exit step (last element) for each answer token
                    step_idx = -1  # the last element in history lists
                    token_confs = []
                    for offset in range(len(ans_token_ids)):
                        pos = rel_pos + offset  # relative to generated segment
                        if pos in conf_hist and len(conf_hist[pos]):
                            token_confs.append(conf_hist[pos][step_idx])
                    if token_confs:
                        confs.append(float(np.mean(token_confs)))
        acc = correct / len(questions)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        accuracy_per_step.append(acc)
        confidence_per_step.append(mean_conf)
        print(f"Step={steps:3d} | Acc={acc:.3f} | Conf={mean_conf:.3f}")

    return accuracy_per_step, confidence_per_step



if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "GSAI-ML/LLaDA-8B-Instruct"  # adjust if necessary

    print("Loading model…")
    model = (
        AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # build constraints dict now that tokenizer is ready
    CONSTRAINTS = _parse_constraints(CONSTRAINTS_TEXT, tokenizer)

    print("Loading GSM8K dataset (main split)…")
    ds = load_dataset("gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    ds_sample = ds.select(indices)

    questions = [QUERY_TEMPLATE.format(Question=item["question"]) for item in ds_sample]
    gt_answers = [extract_answer_from_solution(item["answer"]) for item in ds_sample]
    steps_list = [args.gen_length]
    print(f"Evaluating on {len(questions)} samples across steps: {steps_list}")
    evaluate_on_steps(
        model, tokenizer, questions, gt_answers, 
                            steps_list, 
                            gen_length=args.gen_length, 
                            device=device, 
                            block_length=args.blocklen
    )

