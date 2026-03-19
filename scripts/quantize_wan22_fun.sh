#!/bin/bash
# Quantization script for Wan2.2-Fun-5B
# Converts original checkpoints to quantized format with INT8 linear layers
# and fast Triton-based normalization kernels.

export PYTHONPATH=turbodiffusion

# ============================================================
# Wan2.2-Fun-5B: High-noise model (without quantization)
# ============================================================
python turbodiffusion/inference/quantize_wan22_fun.py \
    --input_path checkpoints/Wan2.2-Fun-5B/merged_high_noise.pth \
    --output_path checkpoints/modified/Wan2.2-Fun-5B-high-720P.pth \
    --attention_type sagesla

# ============================================================
# Wan2.2-Fun-5B: High-noise model (with INT8 quantization)
# ============================================================
python turbodiffusion/inference/quantize_wan22_fun.py \
    --input_path checkpoints/Wan2.2-Fun-5B/merged_high_noise.pth \
    --output_path checkpoints/modified/Wan2.2-Fun-5B-high-720P-quant.pth \
    --attention_type sagesla \
    --quant_linear

# ============================================================
# Wan2.2-Fun-5B: Low-noise model (without quantization)
# ============================================================
python turbodiffusion/inference/quantize_wan22_fun.py \
    --input_path checkpoints/Wan2.2-Fun-5B/merged_low_noise.pth \
    --output_path checkpoints/modified/Wan2.2-Fun-5B-low-720P.pth \
    --attention_type sagesla

# ============================================================
# Wan2.2-Fun-5B: Low-noise model (with INT8 quantization)
# ============================================================
python turbodiffusion/inference/quantize_wan22_fun.py \
    --input_path checkpoints/Wan2.2-Fun-5B/merged_low_noise.pth \
    --output_path checkpoints/modified/Wan2.2-Fun-5B-low-720P-quant.pth \
    --attention_type sagesla \
    --quant_linear

# ============================================================
# Alternative: Load from safetensors format (e.g., HuggingFace)
# ============================================================
# python turbodiffusion/inference/quantize_wan22_fun.py \
#     --input_path checkpoints/Wan2.2-Fun-5B/diffusion_pytorch_model.safetensors \
#     --output_path checkpoints/modified/Wan2.2-Fun-5B-high-720P-quant.pth \
#     --input_format safetensors \
#     --attention_type sagesla \
#     --quant_linear
