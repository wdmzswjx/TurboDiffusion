#!/bin/bash
# Inference script for Wan2.2-Fun-5B with INT8 quantization
#
# VRAM requirements (approximate):
#   - Without quantization: ~12 GB (single model on GPU)
#   - With INT8 quantization: ~7 GB (single model on GPU)
#   - CPU offloading: only one model on GPU at a time

export PYTHONPATH=turbodiffusion

# ============================================================
# Wan2.2-Fun-5B: I2V inference with quantization + VRAM optimization
# ============================================================
python turbodiffusion/inference/wan2.2_fun_infer.py \
    --image_path input.jpg \
    --high_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-high-720P-quant.pth \
    --low_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-low-720P-quant.pth \
    --prompt "A beautiful sunset over the ocean, waves gently rolling" \
    --quant_linear \
    --attention_type sagesla \
    --num_steps 4 \
    --resolution 720p \
    --aspect_ratio 16:9 \
    --save_path output/wan22_fun_5b_quant.mp4

# ============================================================
# Without quantization (higher quality, more VRAM)
# ============================================================
# python turbodiffusion/inference/wan2.2_fun_infer.py \
#     --image_path input.jpg \
#     --high_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-high-720P.pth \
#     --low_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-low-720P.pth \
#     --prompt "A cat walking on the beach" \
#     --attention_type sagesla \
#     --num_steps 4 \
#     --save_path output/wan22_fun_5b.mp4

# ============================================================
# Adaptive resolution (match input image aspect ratio)
# ============================================================
# python turbodiffusion/inference/wan2.2_fun_infer.py \
#     --image_path input.jpg \
#     --high_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-high-720P-quant.pth \
#     --low_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-low-720P-quant.pth \
#     --prompt "Dynamic camera movement through a forest" \
#     --quant_linear \
#     --attention_type sagesla \
#     --adaptive_resolution \
#     --num_steps 4 \
#     --save_path output/wan22_fun_5b_adaptive.mp4

# ============================================================
# Fast 1-step generation (lowest quality, fastest)
# ============================================================
# python turbodiffusion/inference/wan2.2_fun_infer.py \
#     --image_path input.jpg \
#     --high_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-high-720P-quant.pth \
#     --low_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-low-720P-quant.pth \
#     --prompt "A person smiling" \
#     --quant_linear \
#     --attention_type sagesla \
#     --num_steps 1 \
#     --ode \
#     --save_path output/wan22_fun_5b_fast.mp4
