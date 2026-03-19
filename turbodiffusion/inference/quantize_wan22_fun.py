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
import torch.nn as nn

from modify_model import (
    select_model,
    replace_attention,
    tensor_kwargs,
)
from rcm.utils.model_utils import load_state_dict
from rcm.networks.wan2pt2 import (
    WanRMSNorm as WanRMSNorm2pt2,
    WanLayerNorm as WanLayerNorm2pt2,
)
from ops import Int8Linear, FastRMSNorm, FastLayerNorm


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


def remap_state_dict(state_dict: dict, net: nn.Module, prefix: str = "") -> dict:
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


def replace_linear_norm_full(
    model: nn.Module,
    replace_linear: bool = False,
    replace_norm: bool = False,
    quantize: bool = True,
    skip_patterns: list = None,
) -> nn.Module:
    """
    Replace Linear and Norm layers across the ENTIRE model (not just model.blocks).

    Unlike the original replace_linear_norm which only operates on model.blocks,
    this version traverses the full model graph. Certain sensitive layers
    (e.g., patch_embedding, time_projection) can be skipped via skip_patterns.

    Args:
        model: The full WanModel instance.
        replace_linear: Whether to replace nn.Linear with Int8Linear.
        replace_norm: Whether to replace WanRMSNorm/WanLayerNorm with fast versions.
        quantize: Whether to actually quantize weights (True) or just create
                  Int8Linear shells for later loading (False).
        skip_patterns: List of name substrings to skip (e.g., ["patch_embedding"]).
    """
    if skip_patterns is None:
        # Skip layers where quantization may hurt quality or is unnecessary:
        # - patch_embedding: small, input-facing, important for spatial fidelity
        # - time_projection/time_embedding: small, conditioning pathway
        # - head.head: output layer, quality-sensitive
        # - proj_l: projection layers skipped in original implementation
        skip_patterns = ["patch_embedding", "head.head", "proj_l"]

    replacements = {}
    quant_count = 0
    skip_count = 0

    for name, module in model.named_modules():
        should_skip = any(pat in name for pat in skip_patterns)

        if isinstance(module, nn.Linear) and replace_linear:
            if should_skip:
                skip_count += 1
                print(f"  [SKIP] {name}: {module.weight.shape} (matches skip pattern)")
            else:
                replacements[name] = Int8Linear.from_linear(module, quantize)
                quant_count += 1

        if isinstance(module, WanRMSNorm2pt2) and replace_norm:
            if not should_skip:
                replacements[name] = FastRMSNorm.from_rmsnorm(module)

        if isinstance(module, WanLayerNorm2pt2) and replace_norm:
            if not should_skip:
                replacements[name] = FastLayerNorm.from_layernorm(module)

    # Apply replacements via setattr on parent modules
    for name, new_module in replacements.items():
        name_parts = name.split(".")
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, name_parts[-1], new_module)

    if replace_linear:
        print(f"  Quantized {quant_count} Linear layers, skipped {skip_count}")
    return model


def diagnose_state_dict(model: nn.Module, label: str = "Model"):
    """Print detailed diagnostics of state_dict contents for debugging."""
    sd = model.state_dict()

    total_bytes = 0
    int8_bytes = 0
    bf16_bytes = 0
    fp32_bytes = 0
    other_bytes = 0
    int8_count = 0
    linear_count = 0

    dtype_sizes = {
        torch.int8: 1, torch.float16: 2, torch.bfloat16: 2,
        torch.float32: 4, torch.float64: 8, torch.int32: 4, torch.int64: 8,
    }

    for key, tensor in sd.items():
        nbytes = tensor.numel() * dtype_sizes.get(tensor.dtype, tensor.element_size())
        total_bytes += nbytes
        if tensor.dtype == torch.int8:
            int8_bytes += nbytes
            int8_count += 1
        elif tensor.dtype == torch.bfloat16:
            bf16_bytes += nbytes
        elif tensor.dtype == torch.float32:
            fp32_bytes += nbytes
        else:
            other_bytes += nbytes

    for module in model.modules():
        if isinstance(module, Int8Linear):
            linear_count += 1

    print(f"\n{'='*60}")
    print(f"  Diagnostics: {label}")
    print(f"{'='*60}")
    print(f"  Total state_dict size:  {total_bytes / 1024**2:.1f} MB")
    print(f"    - int8 tensors:       {int8_bytes / 1024**2:.1f} MB ({int8_count} tensors)")
    print(f"    - bf16 tensors:       {bf16_bytes / 1024**2:.1f} MB")
    print(f"    - fp32 tensors:       {fp32_bytes / 1024**2:.1f} MB")
    print(f"    - other:              {other_bytes / 1024**2:.1f} MB")
    print(f"  Int8Linear modules:     {linear_count}")
    print(f"  nn.Linear modules:      {sum(1 for m in model.modules() if type(m) is nn.Linear)}")
    if int8_count == 0 and linear_count == 0:
        print(f"  *** WARNING: No INT8 quantization detected! ***")
        print(f"  *** Did you forget --quant_linear? ***")
    print(f"{'='*60}\n")

    return total_bytes


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
        "--skip_patterns", type=str, nargs="*",
        default=["patch_embedding", "head.head", "proj_l"],
        help="Layer name patterns to skip during quantization"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only analyze the model without saving"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not args.quant_linear:
        print("WARNING: --quant_linear not set. No Linear layers will be quantized!")
        print("         The output model will be the same size as the input.")
        print("         Add --quant_linear to enable INT8 quantization.\n")

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

    # Diagnose BEFORE quantization
    print("\n--- Before quantization ---")
    original_bytes = diagnose_state_dict(net, "Original Model")

    # Apply quantization using the full-model version
    net = replace_linear_norm_full(
        net,
        replace_linear=args.quant_linear,
        replace_norm=not args.default_norm,
        quantize=True,
        skip_patterns=args.skip_patterns,
    )

    # Diagnose AFTER quantization
    print("\n--- After quantization ---")
    quantized_bytes = diagnose_state_dict(net, "Quantized Model")

    if original_bytes > 0:
        ratio = quantized_bytes / original_bytes
        print(f"  Size ratio: {ratio:.2f}x ({(1-ratio)*100:.1f}% reduction)")

    if args.dry_run:
        print("Dry run complete. No model saved.")
    else:
        print(f"[5/5] Saving quantized model to: {args.output_path}")
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        torch.save(net.state_dict(), args.output_path)
        file_size_mb = os.path.getsize(args.output_path) / (1024 ** 2)
        print(f"  Saved file size: {file_size_mb:.1f} MB")
        print("Done!")
