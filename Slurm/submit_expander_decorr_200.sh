#!/usr/bin/env bash
set -euo pipefail

# Two configurations, three independent seeds, 200 epochs per run.
SEEDS=(1407 2407 3407)

for seed in "${SEEDS[@]}"; do
  sbatch --export="ALL,EXPERIMENT_NAME=expander_decorr_200_exp1_s${seed},TRAIN_ARGS=--n-epochs-constant 100 --n-epochs-cosine 100 --lambda-expander-decorr 0.0001 --lambda-expander-decorr-var 0.0025 --expander-decorr-warmup-epochs 20 --seed ${seed}" \
    Slurm/train.slurm

  sbatch --export="ALL,EXPERIMENT_NAME=expander_decorr_200_exp4_s${seed},TRAIN_ARGS=--n-epochs-constant 100 --n-epochs-cosine 100 --lambda-expander-decorr 0.1 --lambda-expander-decorr-var 0.25 --expander-decorr-warmup-epochs 20 --seed ${seed}" \
    Slurm/train.slurm
done

echo "Submitted 6 jobs: exp1/exp4 x seeds ${SEEDS[*]}"