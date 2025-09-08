# Prophet: Fast Decoding for Diffusion Language Models

Official implementation of "Diffusion Language Models Know the Answer Before Decoding"

## Overview
Prophet is a training-free early-exit decoding paradigm for Diffusion Language Models (DLMs) that leverages the observation that correct answers often emerge early in the decoding process, well before the final step.
<img width="1938" height="824" alt="Image" src="https://github.com/user-attachments/assets/972eb05c-c3fd-4b21-a2a4-ce50f0045b73" />

## Key Features

- **Training-free**: No additional training required, works directly with existing DLMs
- **Dynamic early-exit**: Uses confidence gap between top-2 predictions as stopping criterion
- **Significant speedup**: Up to 2.67× faster on planning tasks, 2.34× on general tasks
- **Quality preservation**: Maintains or improves generation quality compared to full decoding
- **Model-agnostic**: Compatible with different DLMs (tested on LLaDA-8B and Dream-7B)

## Installation

### Requirements
```bash
pip install torch>=2.0.0 transformers==4.38.2 accelerate datasets tqdm
```

### Quick Start
```bash
git clone https://github.com/pixeli99/Prophet.git
cd Prophet
```

## Usage

### Basic Generation with Prophet Early Exit

```python
import torch
from transformers import AutoModel, AutoTokenizer
from generate_earlyexit import generate

# Load model
model = AutoModel.from_pretrained(
    'GSAI-ML/LLaDA-8B-Instruct', 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(
    'GSAI-ML/LLaDA-8B-Instruct', 
    trust_remote_code=True
)

# Prepare prompt
prompt = "What is the capital of France?"
messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(prompt_text, return_tensors="pt")['input_ids'].cuda()

# Generate with Prophet early exit
output, gap_data = generate(
    model, 
    input_ids,
    steps=256,
    gen_length=256,
    block_length=32,
    analyze_gap=True,
    answer_start_pos=input_ids.shape[1] + 200,  # Estimated answer position
    early_exit_thresholds={'early': 7.5, 'mid': 5.0, 'late': 2.5}
)

# Decode output
generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
print(f"Generated: {generated_text}")
print(f"Early exit: {gap_data['exit_info']['early_exit_triggered']} at step {gap_data['exit_info']['exit_decision_step']}")
```

## Evaluation

### Running Benchmarks

We provide evaluation scripts compatible with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

#### GSM8K Evaluation with Prophet
```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch eval_llada.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --model_args model_path='/path/to/LLaDA-8B-Instruct',enable_early_exit=true,constraints_text="200:The|201:answer|202:is",gen_length=256,steps=256,block_length=32
```

#### Other Benchmarks
```bash
# MMLU
accelerate launch eval_llada.py \
  --tasks mmlu \
  --model llada_dist \
  --model_args model_path='/path/to/LLaDA-8B-Instruct',enable_early_exit=true,gen_length=128,steps=128

# HumanEval
accelerate launch eval_llada.py \
  --tasks humaneval \
  --model llada_dist \
  --model_args model_path='/path/to/LLaDA-8B-Instruct',enable_early_exit=true,gen_length=512,steps=512
```

### Configuration Parameters

#### Prophet Early Exit Parameters
- `enable_early_exit`: Enable Prophet early exit mechanism (default: false)
- `early_threshold`: Gap threshold for early phase (0-33% progress, default: 7.5)
- `mid_threshold`: Gap threshold for middle phase (33-67% progress, default: 5.0)  
- `late_threshold`: Gap threshold for late phase (67-100% progress, default: 2.5)
- `constraints_text`: Constraint tokens to force generation (e.g., "200:The|201:answer|202:is")
- `answer_length`: Number of answer tokens to monitor (default: 5)

#### Generation Parameters
- `steps`: Total diffusion steps (default: 256)
- `gen_length`: Maximum generation length (default: 256)
- `block_length`: Block size for semi-autoregressive generation (default: 32)
- `temperature`: Sampling temperature (default: 0.0)
- `cfg_scale`: Classifier-free guidance scale (default: 0.0)
- `remasking`: Remasking strategy ('low_confidence' or 'random', default: 'low_confidence')

## Core Components

### `generate_earlyexit.py`
Main generation function with Prophet early exit mechanism:
- `generate()`: LLaDA generation with logits gap monitoring
- `should_early_exit()`: Phase-aware exit decision logic

### `generate.py`
Baseline LLaDA generation without early exit for comparison.

### `eval_llada.py`
Evaluation harness integration with Prophet support.

## Citation

If you use Prophet in your research, please cite:

```bibtex
@article{li2025diffusion,
  title={Diffusion Language Models Know the Answer Before Decoding},
  author={Li, Pengxiang and Zhou, Yefan and Muhtar, Dilxat and Yin, Lu and Yan, Shilin and Shen, Li and Liang, Yi and Vosoughi, Soroush and Liu, Shiwei},
  journal={arXiv preprint arXiv:2508.19982},
  year={2025}
}
```

## Acknowledgments

This implementation is based on the [LLaDA](https://github.com/ML-GSAI/LLaDA) model architecture. We thank the LLaDA team for open-sourcing their models and code.