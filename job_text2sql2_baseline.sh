#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=TC2N03,TC2N04,TC2N05,TC2N06,TC2N07,TC2N08
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=text2sql2-baseline
#SBATCH --output=output_text2sql2_%x_%j.out
#SBATCH --error=error_text2sql2_%x_%j.err

# Baseline：Text2SQL2.py（无 plan / repair），Spider test 前 500 条，seed=42
# 用法:
#   cd /home/msai/junjie012/nlp && sbatch job_text2sql2_baseline.sh
# 评 dev：在下方命令加 --split dev

set -euo pipefail

NLP_ROOT="/home/msai/junjie012/nlp"
VENV="${NLP_ROOT}/.venv"

echo "Running on: $(hostname)"
date
nvidia-smi

source "${VENV}/bin/activate"
cd "${NLP_ROOT}"

export CUDA_VISIBLE_DEVICES=0

python -c "import torch, sentence_transformers, transformers, accelerate, faiss, numpy, nltk" >/dev/null 2>&1 || {
  echo "Python dependencies missing in ${VENV}"
  echo "Fix: pip install torch sentence-transformers transformers accelerate faiss-cpu tqdm nltk   (see requirements_ft.txt)"
  exit 2
}

python Text2SQL2.py --seed 42 --split test --eval-samples 500

echo "Done."
date
