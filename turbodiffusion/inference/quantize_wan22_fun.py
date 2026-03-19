"""
Quantization script for Wan2.2-Fun-5B model.

This script loads a Wan2.2-Fun-5B checkpoint (from safetensors or pth),
replaces Linear layers with Int8Linear, optionally replaces norms with
fast Triton-based versions, and saves the quantized model for inference.

Usage:
    python quantize_wan22_fun.py \
        --input_path checkpoints/Wan2.2-Fun-5B/model.safetensors \
        --output_path checkpoints/modified/Wan2.2-Fun-5B-quant.pth \
        --attention_type sagesla \
        --quant_linear \
        --input_format safetensors
"""

import argparse
import os

import torch

from modify_model import (
    select_model,
    replace_attention,
    replace_linear_norm,
    tensor_kwargs,
)
from rcm.utils.model_utils import load_state_dict


def load_checkpoint(input_path: str, input_format: str) -> dict:
    """Load checkpoint from pth or safetensors format."""
    if input_format == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("Please install safetensors: pip install safetensors")
        state_dict = load_file(input_path)
    elif input_format == "pth":
        raw = load_state_dict(input_path)
        state_dict = raw.get("state_dict", raw)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    return state_dict


def remap_state_dict(state_dict: dict, net: torch.nn.Module, prefix: str = "") -> dict:
    """
    Remap state_dict keys by stripping known prefixes and reshaping
    patch_embedding weights if needed.
    """
    common_prefixes = ["net.", "model.", "module."]
    if prefix:
        common_prefixes.insert(0, prefix)

    remapped = {}
    for k, v in state_dict.items():
        new_key = k
        for pfx in common_prefixes:
            if new_key.startswith(pfx):
                new_key = new_key[len(pfx):]
                break

        # Reshape patch embedding if shape doesn't match
        if new_key.endswith("patch_embedding.weight"):
            expected_shape = net.patch_embedding.weight.shape
            if v.shape != expected_shape:
                v = v.reshape(expected_shape)
        if new_key.endswith("patch_embedding.bias"):
            expected_shape = net.patch_embedding.bias.shape
            if v.shape != expected_shape:
                v = v.reshape(expected_shape)

        remapped[new_key] = v
    return remapped


def count_parameters(model: torch.nn.Module) -> dict:
    """Count parameters by module type for reporting."""
    total = 0
    quantized = 0
    from ops import Int8Linear
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            numel = param.numel()
            total += numel
            if isinstance(module, Int8Linear):
                quantized += numel
    for name, buf in model.named_buffers():
        if "int8_weight" in name:
            quantized += buf.numel()
    return {"total": total, "quantized_layers_buffers": quantized}


def estimate_vram_savings(model: torch.nn.Module) -> dict:
    """Estimate VRAM savings from INT8 quantization."""
    bf16_size = 0
    int8_size = 0
    from ops import Int8Linear
    for module in model.modules():
        if isinstance(module, Int8Linear):
            # INT8 weight: out*in bytes, scale: row_blocks*col_blocks*4 bytes
            w_bytes = module.int8_weight.numel() * 1  # int8 = 1 byte
            s_bytes = module.scale.numel() * 4  # float32 = 4 bytes
            int8_size += w_bytes + s_bytes
            # Original would be out*in*2 bytes (bf16)
            bf16_size += module.in_features * module.out_features * 2
        elif isinstance(module, torch.nn.Linear):
            bf16_size += module.weight.numel() * 2

    return {
        "original_bf16_mb": bf16_size / (1024 ** 2),
        "quantized_mb": int8_size / (1024 ** 2),
        "savings_mb": (bf16_size - int8_size) / (1024 ** 2),
        "compression_ratio": bf16_size / max(int8_size, 1),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Wan2.2-Fun-5B model for accelerated inference"
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input model checkpoint"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the quantized model checkpoint"
    )
    parser.add_argument(
        "--input_format", choices=["pth", "safetensors"], default="pth",
        help="Format of input checkpoint"
    )
    parser.add_argument(
        "--key_prefix", type=str, default="",
        help="Key prefix to strip from state_dict (e.g. 'net.' or 'model.')"
    )
    parser.add_argument(
        "--attention_type", choices=["sla", "sagesla", "original"], default="sagesla",
        help="Type of attention mechanism to use"
    )
    parser.add_argument(
        "--sla_topk", type=float, default=0.1,
        help="Top-k ratio for SLA/SageSLA attention"
    )
    parser.add_argument(
        "--quant_linear", action="store_true",
        help="Replace Linear layers with INT8 quantized versions"
    )
    parser.add_argument(
        "--default_norm", action="store_true",
        help="Keep default norms (skip fast Triton norm replacement)"
    )
    parser.add_argument(
        "--skip_layer", type=str, default="proj_l",
        help="Layer name pattern to skip during quantization"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only analyze the model without saving"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    print(f"[1/5] Creating Wan2.2-Fun-5B model on meta device...")
    with torch.device("meta"):
        net = select_model("Wan2.2-Fun-5B")

    print(f"[2/5] Loading checkpoint from: {args.input_path} (format: {args.input_format})")
    state_dict = load_checkpoint(args.input_path, args.input_format)
    state_dict = remap_state_dict(state_dict, net, prefix=args.key_prefix)

    # Replace attention before loading weights (attention modules have no learned params to replace)
    if args.attention_type in ["sla", "sagesla"]:
        print(f"[3/5] Replacing attention with {args.attention_type} (topk={args.sla_topk})...")
        net = replace_attention(net, attention_type=args.attention_type, sla_topk=args.sla_topk)
    else:
        print(f"[3/5] Keeping original attention mechanism.")

    # Load weights onto device
    print(f"[4/5] Loading state_dict and applying quantization...")
    net.load_state_dict(state_dict, strict=False, assign=True)
    net = net.to(tensor_kwargs["device"]).eval()
    del state_dict

    # Apply quantization and fast norm replacement AFTER loading weights
    net = replace_linear_norm(
        net,
        replace_linear=args.quant_linear,
        replace_norm=not args.default_norm,
        quantize=True,
        skip_layer=args.skip_layer,
    )

    # Report VRAM savings
    savings = estimate_vram_savings(net)
    print(f"  - Original BF16 weight size: {savings['original_bf16_mb']:.1f} MB")
    print(f"  - Quantized weight size:     {savings['quantized_mb']:.1f} MB")
    print(f"  - VRAM savings:              {savings['savings_mb']:.1f} MB")
    print(f"  - Compression ratio:         {savings['compression_ratio']:.2f}x")

    if args.dry_run:
        print("Dry run complete. No model saved.")
    else:
        print(f"[5/5] Saving quantized model to: {args.output_path}")
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        torch.save(net.state_dict(), args.output_path)
        file_size_mb = os.path.getsize(args.output_path) / (1024 ** 2)
        print(f"  - Saved file size: {file_size_mb:.1f} MB")
        print("Done!")
