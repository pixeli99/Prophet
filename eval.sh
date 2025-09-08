export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0,1
#  --limit 256

accelerate launch eval_llada.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',enable_early_exit=true,constraints_text="200:The|201:answer|202:is",gen_length=256,steps=256,block_length=32,answer_length=5 \