#!/bin/bash
# =================================================================
# End-to-end test pipeline for Wan2.2-Fun-5B I2V TurboDiffusion
#
# Workflow:
#   1. Convert DCP checkpoint → pth
#   2. Merge rCM student into base model
#   3. Quantize (INT8 + fast norms + SLA attention)
#   4. Run inference
# =================================================================

export PYTHONPATH=turbodiffusion

# =================================================================
# Step 1: Convert DCP checkpoint to pth
# =================================================================
echo "=== Step 1: DCP → PTH Conversion ==="

# High-noise model (from rCM distillation)
python turbodiffusion/scripts/dcp_to_pth.py \
    --input_path outputs/rcm_i2v/checkpoints/iter_100000 \
    --output_path checkpoints/Wan2.2-Fun-5B/rcm_student_raw.pth

# =================================================================
# Step 2: Merge student model with base model
# =================================================================
echo "=== Step 2: Model Merging ==="

# Merge for high-noise regime
python turbodiffusion/scripts/merge_models.py \
    --base_path checkpoints/Wan2.2-Fun-5B/base_model.pth \
    --diff_target_path checkpoints/Wan2.2-Fun-5B/rcm_student_raw.pth \
    --diff_base_path checkpoints/Wan2.2-Fun-5B/base_model.pth \
    --output_path checkpoints/Wan2.2-Fun-5B/merged_high_noise.pth \
    --weight 1.0

# Repeat for low-noise model (using appropriate checkpoint)
# python turbodiffusion/scripts/merge_models.py \
#     --base_path checkpoints/Wan2.2-Fun-5B/base_model.pth \
#     --diff_target_path checkpoints/Wan2.2-Fun-5B/rcm_student_low_raw.pth \
#     --diff_base_path checkpoints/Wan2.2-Fun-5B/base_model.pth \
#     --output_path checkpoints/Wan2.2-Fun-5B/merged_low_noise.pth \
#     --weight 1.0

# =================================================================
# Step 3: Quantize + Replace attention + Replace norms
# =================================================================
echo "=== Step 3: Quantization ==="

# Quantize high-noise model
python turbodiffusion/inference/quantize_wan22_fun.py \
    --input_path checkpoints/Wan2.2-Fun-5B/merged_high_noise.pth \
    --output_path checkpoints/modified/Wan2.2-Fun-5B-high-720P-quant.pth \
    --attention_type sagesla \
    --quant_linear

# Quantize low-noise model
python turbodiffusion/inference/quantize_wan22_fun.py \
    --input_path checkpoints/Wan2.2-Fun-5B/merged_low_noise.pth \
    --output_path checkpoints/modified/Wan2.2-Fun-5B-low-720P-quant.pth \
    --attention_type sagesla \
    --quant_linear

# =================================================================
# Step 4: Inference
# =================================================================
echo "=== Step 4: Inference ==="

python turbodiffusion/inference/wan2.2_fun_infer.py \
    --image_path assets/test_images/example.jpg \
    --high_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-high-720P-quant.pth \
    --low_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-low-720P-quant.pth \
    --prompt "A cat walking gracefully on a sunny beach" \
    --model Wan2.2-Fun-5B \
    --quant_linear \
    --attention_type sagesla \
    --num_steps 4 \
    --resolution 720p \
    --aspect_ratio 16:9 \
    --save_path output/test_wan22_fun_5b.mp4

echo "=== Pipeline Complete ==="
echo "Output saved to output/test_wan22_fun_5b.mp4"
