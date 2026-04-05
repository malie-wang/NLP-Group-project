#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=TC2N03,TC2N04,TC2N05,TC2N06,TC2N07,TC2N08
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=text2sql
#SBATCH --output=logs/slurm/output_text2sql_%x_%j.out
#SBATCH --error=logs/slurm/error_text2sql_%x_%j.err

# 用法（在登录节点）:
#   cd /home/msai/junjie012/nlp && sbatch job_text2sql.sh
# 可选：覆盖模型 / 加载 LoRA 做 EX 评测（adapter 须为有效权重，非空文件）
#   sbatch --export=ALL,CHAT_LORA_PATH=trained_models/qwen3-4b-spider-cot-lora job_text2sql.sh
#   sbatch --export=ALL,EMBED_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B,CHAT_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 job_text2sql.sh

set -euo pipefail

NLP_ROOT="/home/msai/junjie012/nlp"
VENV="${NLP_ROOT}/.venv"

mkdir -p "${NLP_ROOT}/logs/slurm"

echo "Running on: $(hostname)"
date
nvidia-smi

source "${VENV}/bin/activate"
cd "${NLP_ROOT}"

export CUDA_VISIBLE_DEVICES=0

python -c "import torch, sentence_transformers, transformers, accelerate, faiss, numpy, dotenv, nltk" >/dev/null 2>&1 || {
  echo "Python dependencies missing in ${VENV}"
  echo "Fix: pip install torch sentence-transformers transformers accelerate faiss-cpu python-dotenv tqdm nltk   (see requirements_ft.txt)"
  exit 2
}

if [ -n "${CHAT_LORA_PATH:-}" ]; then
  python -c "import peft" >/dev/null 2>&1 || {
    echo "CHAT_LORA_PATH is set but peft is missing in ${VENV}"
    echo "Fix: pip install peft   (see requirements_ft.txt)"
    exit 2
  }
fi

# Spider test, 500, seed=42; trace -> cache/eval_trace_test.jsonl; logs -> logs/slurm/
# 评 dev：加 --split dev；LoRA：sbatch --export=ALL,CHAT_LORA_PATH=... job_text2sql.sh
python Text2SQL.py --seed 42 --split test --eval-samples 500

echo "Done."
date
