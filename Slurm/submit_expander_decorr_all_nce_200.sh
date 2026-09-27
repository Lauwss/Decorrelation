#!/usr/bin/env bash
set -euo pipefail

# Exp4 weights applied to all five PatchNCE layers, three independent seeds.
SEEDS=(1407 2407 3407)
DECORR_LAYERS='stem,downsample_1,downsample_2,enc_resblock_1,enc_resblock_6'

for seed in "${SEEDS[@]}"; do
  sbatch --export="ALL,EXPERIMENT_NAME=expander_decorr_200_exp4_all_nce_s${seed},TRAIN_ARGS=--n-epochs-constant 100 --n-epochs-cosine 100 --lambda-expander-decorr 0.1 --lambda-expander-decorr-var 0.25 --expander-decorr-layers ${DECORR_LAYERS} --expander-decorr-warmup-epochs 20 --seed ${seed}" \
    Slurm/train.slurm
done

echo "Submitted 3 jobs: Exp4 all-PatchNCE-layer decorrelation x seeds ${SEEDS[*]}"