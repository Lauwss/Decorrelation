#!/bin/bash
#SBATCH --job-name=cut-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpushare:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-dehaze}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/ckpt/cut_baseline}"
TRAIN_LOG="${TRAIN_LOG:-$OUTPUT_DIR/train.log}"

mkdir -p "$OUTPUT_DIR"

if [[ -f /etc/profile.d/00-data-shared-cache.sh ]]; then
  source /etc/profile.d/00-data-shared-cache.sh
else
  CACHE_ROOT="/fast/home/$USER/.cache"
  export TMPDIR="$CACHE_ROOT/tmp"
  export PIP_CACHE_DIR="$CACHE_ROOT/pip"
  export TORCH_HOME="$CACHE_ROOT/torch"
  export CUDA_CACHE_PATH="$CACHE_ROOT/nv/ComputeCache"
  export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
  mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" \
    "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
fi
source /data/conda/etc/profile.d/conda.sh

read -r -a EXTRA_ARGS <<< "${TRAIN_ARGS:-}"

cd "$PROJECT_DIR"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "project_dir=$PROJECT_DIR"
  echo "conda_env=$CONDA_ENV"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  conda run --no-capture-output -n "$CONDA_ENV" python train.py \
    --data-root "${DATA_ROOT:-$PROJECT_DIR/datasets/RRSHID}" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --num-workers "${NUM_WORKERS:-8}" \
    "${EXTRA_ARGS[@]}"
} > "$TRAIN_LOG" 2>&1