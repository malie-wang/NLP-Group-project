#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=TC2N03,TC2N04,TC2N05,TC2N06,TC2N07,TC2N08
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --job-name=text2sql_ft
#SBATCH --output=output_ft_text2sql_%x_%j.out
#SBATCH --error=error_ft_text2sql_%x_%j.err

# Text2SQL CoT LoRA — 仅用 nlp/ 内脚本与 venv，不依赖 AI6130_Assignment2。
#
# 首次：
#   cd /home/msai/junjie012/nlp && source .venv/bin/activate && pip install -r requirements_ft.txt
#   python build_cot_ft_dataset.py to-alpaca -i data/sft_messages_524.jsonl -o data/spider_alpaca_524.json
#
# 提交：
#   cd /home/msai/junjie012/nlp && sbatch job_ft_text2sql.sh

set -euo pipefail

NLP_ROOT="/home/msai/junjie012/nlp"
DATA_JSONL="${NLP_ROOT}/qwen3_finetune_sharegpt.jsonl"

echo "Running on: $(hostname)"
date
nvidia-smi

if [[ ! -f "$DATA_JSONL" ]]; then
  echo "ERROR: missing $DATA_JSONL"
  exit 2
fi

source "${NLP_ROOT}/.venv/bin/activate"
cd "${NLP_ROOT}"

export CUDA_VISIBLE_DEVICES=0
# 减轻显存碎片（OOM 时可与 fp16_full_eval 一起用）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch, transformers, peft, datasets" >/dev/null 2>&1 || {
  echo "Missing deps. Run: pip install -r ${NLP_ROOT}/requirements_ft.txt"
  exit 2
}

python finetune_text2sql.py \
  --base_model "Qwen/Qwen3-4B-Instruct-2507" \
  --data_path "$DATA_JSONL" \
  --data_format sharegpt \
  --output_dir "${NLP_ROOT}/trained_models/qwen3-4b-sharegpt-cot-lora" \
  --batch_size 8 \
  --micro_batch_size 1 \
  --num_epochs 2 \
  --learning_rate 2e-4 \
  --cutoff_len 3072 \
  --val_set_size 52 \
  --eval_steps 50 \
  --save_steps 50 \
  --per_device_eval_batch_size 1

echo "Done."
date
