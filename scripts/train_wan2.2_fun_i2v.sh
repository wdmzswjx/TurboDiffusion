#!/bin/bash
# =================================================================
# Training script for Wan2.2-Fun-5B I2V TurboDiffusion
#
# Two training stages:
#   1. SLA (Sparse Linear Attention) distillation
#   2. rCM (Rectified Consistency Model) distillation
#
# Prerequisites:
#   - Pre-process data into WebDataset tar shards with:
#     latent.pt, embed.pt, prompt.txt, image_latent.pt (optional)
#   - Teacher checkpoint in DCP format at assets/checkpoints/
#   - VAE and text encoder weights
# =================================================================

export PYTHONPATH=turbodiffusion

NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-12341}

# =================================================================
# Stage 1: SLA distillation (simpler, faster convergence)
# =================================================================
echo "=== Stage 1: SLA Distillation ==="

# Debug run (quick sanity check)
# torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
#     -m scripts.train --config=rcm/configs/registry_sla_i2v.py -- \
#     experiment="wan2pt2_fun_5B_i2v_SLA_debug"

# Full training run
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    -m scripts.train --config=rcm/configs/registry_sla_i2v.py -- \
    experiment="wan2pt2_fun_5B_i2v_SLA" \
    dataloader_train.tar_path_pattern="assets/datasets/Wan2.2_Fun_5B_720p_I2V/shard*.tar"

# =================================================================
# Stage 2: rCM distillation (higher quality, needs fake_score)
# =================================================================
echo "=== Stage 2: rCM Distillation ==="

# Debug run
# torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
#     -m scripts.train --config=rcm/configs/registry_distill_i2v.py -- \
#     experiment="wan2pt2_fun_5B_i2v_rCM_debug"

# Full training run
# Optionally load SLA-trained checkpoint as starting point:
#   checkpoint.load_path="outputs/sla_i2v/checkpoints/iter_XXXXX"
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    -m scripts.train --config=rcm/configs/registry_distill_i2v.py -- \
    experiment="wan2pt2_fun_5B_i2v_rCM" \
    dataloader_train.tar_path_pattern="assets/datasets/Wan2.2_Fun_5B_720p_I2V/shard*.tar"
