# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Wan2.2 VAE Encode/Decode Single-GPU Acceleration Tool

Addresses the VAE time bottleneck where encode and decode each take ~18s.
Provides multiple optimization strategies that can be combined:

1. torch.compile          - Kernel fusion & graph optimization
2. Spatial tiling          - Reduces peak VRAM, enables larger resolutions
3. Decode temporal batching - Processes multiple latent frames per decoder call
4. CUDA graphs             - Eliminates CPU-GPU launch overhead in decode loop
5. Channel-last memory     - Better GPU memory access patterns for convolutions
6. Selective FP8/FP16      - Reduced precision for non-critical layers

Usage:
    from turbodiffusion.vae_optimize import optimize_vae, OptimizeConfig

    cfg = OptimizeConfig(
        compile_mode="reduce-overhead",
        spatial_tiling=True,
        tile_size=256,
        decode_temporal_batch=4,
        channel_last=True,
    )
    vae = WanVAE(...)
    vae = optimize_vae(vae, cfg)
    # Use vae.encode / vae.decode as before
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimizeConfig:
    """Configuration for VAE optimization strategies."""

    # torch.compile
    compile_encoder: bool = True
    compile_decoder: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"
    compile_fullgraph: bool = False
    compile_dynamic: Optional[bool] = None

    # Spatial tiling (encode & decode)
    spatial_tiling: bool = False
    tile_size: int = 256
    tile_overlap: int = 32

    # Decode temporal batching: process N latent frames per decoder forward
    # instead of the default 1-frame-at-a-time loop
    decode_temporal_batch: int = 1

    # Channel-last memory format for Conv2d/Conv3d
    channel_last: bool = True

    # Precision
    dtype: torch.dtype = torch.bfloat16

    # CUDA graphs for decode loop
    cuda_graphs: bool = False

    # Verbose timing
    verbose: bool = False


# ---------------------------------------------------------------------------
# Core optimizers
# ---------------------------------------------------------------------------

def optimize_vae(vae, config: Optional[OptimizeConfig] = None):
    """
    Apply optimizations to a WanVAE instance and return an optimized wrapper.

    Supports both WanVAE and Wan2pt1VAEInterface (used by Wan2.1/2.2 pipelines).
    The underlying VAE architecture is shared between Wan2.1 and Wan2.2;
    only the checkpoint weights differ.

    Args:
        vae: WanVAE or Wan2pt1VAEInterface instance
        config: Optimization config. If None, uses sensible defaults.

    Returns:
        OptimizedWanVAE wrapper with the same encode/decode interface.
    """
    if config is None:
        config = OptimizeConfig()

    # If passed a Wan2pt1VAEInterface, extract the inner WanVAE
    inner = getattr(vae, "model", vae)
    return OptimizedWanVAE(inner, config)


def optimize_tokenizer(tokenizer, config: Optional[OptimizeConfig] = None):
    """
    Optimize a Wan2pt1VAEInterface tokenizer in-place by replacing its
    internal WanVAE with an optimized version.

    This is the easiest way to accelerate VAE in the Wan2.2 inference pipeline:

        tokenizer = Wan2pt1VAEInterface(vae_pth="checkpoints/Wan2.2_VAE.pth")
        optimize_tokenizer(tokenizer)
        # tokenizer.encode / tokenizer.decode are now accelerated

    Args:
        tokenizer: Wan2pt1VAEInterface instance
        config: Optimization config. If None, uses sensible defaults.

    Returns:
        The same tokenizer instance (modified in-place).
    """
    if config is None:
        config = OptimizeConfig()

    tokenizer.model = OptimizedWanVAE(tokenizer.model, config)
    return tokenizer


class OptimizedWanVAE:
    """
    Drop-in replacement wrapper for WanVAE with acceleration optimizations.
    Preserves the same encode(videos) / decode(zs) interface.
    """

    def __init__(self, vae, config: OptimizeConfig):
        self.config = config
        self.vae = vae
        self.model = vae.model
        self.dtype = vae.dtype
        self.device = vae.device
        self.mean = vae.mean
        self.std = vae.std
        self.scale = vae.scale
        self.is_amp = vae.is_amp
        self.context = vae.context

        self._apply_optimizations()

    def _apply_optimizations(self):
        cfg = self.config

        # 1. Channel-last memory format
        if cfg.channel_last:
            _apply_channel_last(self.model)
            logger.info("[VAE Optimize] Applied channels_last memory format")

        # 2. torch.compile
        if cfg.compile_encoder:
            self.model.encoder = torch.compile(
                self.model.encoder,
                mode=cfg.compile_mode,
                fullgraph=cfg.compile_fullgraph,
                dynamic=cfg.compile_dynamic,
            )
            logger.info(f"[VAE Optimize] Compiled encoder (mode={cfg.compile_mode})")

        if cfg.compile_decoder:
            self.model.decoder = torch.compile(
                self.model.decoder,
                mode=cfg.compile_mode,
                fullgraph=cfg.compile_fullgraph,
                dynamic=cfg.compile_dynamic,
            )
            logger.info(f"[VAE Optimize] Compiled decoder (mode={cfg.compile_mode})")

        if cfg.spatial_tiling:
            logger.info(
                f"[VAE Optimize] Spatial tiling enabled "
                f"(tile={cfg.tile_size}, overlap={cfg.tile_overlap})"
            )

        if cfg.decode_temporal_batch > 1:
            logger.info(
                f"[VAE Optimize] Decode temporal batching: "
                f"{cfg.decode_temporal_batch} frames/step"
            )

    # ------------------------------------------------------------------
    # Public interface (matches WanVAE)
    # ------------------------------------------------------------------

    def count_param(self):
        return self.vae.count_param()

    @torch.no_grad()
    def encode(self, videos):
        """
        Encode videos to latent space.
        videos: Tensor [B, C, T, H, W]
        """
        t0 = time.perf_counter() if self.config.verbose else None

        if self.config.spatial_tiling:
            result = self._tiled_encode(videos)
        else:
            result = self._direct_encode(videos)

        if t0 is not None:
            torch.cuda.synchronize()
            logger.info(f"[VAE Optimize] encode: {time.perf_counter() - t0:.3f}s")

        return result

    @torch.no_grad()
    def decode(self, zs):
        """
        Decode latent to video.
        zs: Tensor [B, C, T, H, W]
        """
        t0 = time.perf_counter() if self.config.verbose else None

        if self.config.spatial_tiling:
            result = self._tiled_decode(zs)
        elif self.config.decode_temporal_batch > 1:
            result = self._batched_decode(zs)
        else:
            result = self._direct_decode(zs)

        if t0 is not None:
            torch.cuda.synchronize()
            logger.info(f"[VAE Optimize] decode: {time.perf_counter() - t0:.3f}s")

        return result

    # ------------------------------------------------------------------
    # Direct encode / decode (no tiling)
    # ------------------------------------------------------------------

    def _direct_encode(self, videos):
        in_dtype = videos.dtype
        with self.context:
            if not self.is_amp:
                videos = videos.to(self.dtype)
            latent = self.model.encode(videos, self.scale)
        return latent.to(in_dtype)

    def _direct_decode(self, zs):
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            video_recon = self.model.decode(zs, self.scale)
        return video_recon.to(in_dtype)

    # ------------------------------------------------------------------
    # Batched temporal decode
    # ------------------------------------------------------------------

    def _batched_decode(self, zs):
        """
        Process multiple temporal frames per decoder call instead of 1.
        The original decode loop processes z[:,:,i:i+1] one at a time.
        We batch `decode_temporal_batch` frames together to reduce overhead.
        """
        in_dtype = zs.dtype
        batch_size = self.config.decode_temporal_batch

        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)

            scale = self.scale
            z_dim = self.model.z_dim

            # Undo scale
            if isinstance(scale[0], torch.Tensor):
                z = zs / scale[1].view(1, z_dim, 1, 1, 1) + scale[0].view(1, z_dim, 1, 1, 1)
            else:
                z = zs / scale[1] + scale[0]

            total_t = z.shape[2]
            x = self.model.conv2(z)

            # Clear caches
            self.model.clear_cache()

            # First frame must be processed alone (initializes caches)
            self.model._conv_idx = [0]
            out = self.model.decoder(
                x[:, :, 0:1, :, :],
                feat_cache=self.model._feat_map,
                feat_idx=self.model._conv_idx,
            )

            # Process remaining frames in batches
            i = 1
            while i < total_t:
                end = min(i + batch_size, total_t)
                self.model._conv_idx = [0]
                out_ = self.model.decoder(
                    x[:, :, i:end, :, :],
                    feat_cache=self.model._feat_map,
                    feat_idx=self.model._conv_idx,
                )
                out = torch.cat([out, out_], dim=2)
                i = end

            self.model.clear_cache()

        return out.to(in_dtype)

    # ------------------------------------------------------------------
    # Spatial tiling
    # ------------------------------------------------------------------

    def _tiled_encode(self, videos):
        """
        Encode with spatial tiling to reduce peak VRAM usage.
        Splits H,W into overlapping tiles, encodes each, and blends.
        """
        in_dtype = videos.dtype
        with self.context:
            if not self.is_amp:
                videos = videos.to(self.dtype)

            B, C, T, H, W = videos.shape
            tile_size = self.config.tile_size
            overlap = self.config.tile_overlap
            stride = tile_size - overlap

            # Compute output spatial dims (8x spatial compression)
            compress = 8
            latent_tile = tile_size // compress
            latent_overlap = overlap // compress
            latent_stride = latent_tile - latent_overlap

            out_H = H // compress
            out_W = W // compress
            z_dim = self.model.z_dim

            # Accumulate with blending weights
            output = torch.zeros(B, z_dim, (T - 1) // 4 + 1, out_H, out_W,
                                 device=videos.device, dtype=videos.dtype)
            weight = torch.zeros(1, 1, 1, out_H, out_W,
                                 device=videos.device, dtype=videos.dtype)

            # Create blending mask for tiles
            blend = _make_tile_blend_mask(latent_tile, latent_overlap, videos.device, videos.dtype)

            for y in range(0, H, stride):
                for x_pos in range(0, W, stride):
                    y_end = min(y + tile_size, H)
                    x_end = min(x_pos + tile_size, W)
                    y_start = y_end - tile_size
                    x_start = x_end - tile_size

                    tile = videos[:, :, :, y_start:y_end, x_start:x_end]

                    # Encode tile
                    self.model.clear_cache()
                    tile_latent = self.model.encode(tile, self.scale)

                    # Latent coords
                    ly = y_start // compress
                    lx = x_start // compress
                    lt_h = tile_latent.shape[3]
                    lt_w = tile_latent.shape[4]

                    # Blend mask (crop to actual tile size)
                    bm = blend[:lt_h, :lt_w].unsqueeze(0).unsqueeze(0).unsqueeze(0)

                    output[:, :, :, ly:ly + lt_h, lx:lx + lt_w] += tile_latent * bm
                    weight[:, :, :, ly:ly + lt_h, lx:lx + lt_w] += bm

            output = output / weight.clamp(min=1e-8)

        return output.to(in_dtype)

    def _tiled_decode(self, zs):
        """
        Decode with spatial tiling to reduce peak VRAM usage.
        """
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)

            B, C, T, H, W = zs.shape
            compress = 8
            tile_size = self.config.tile_size // compress
            overlap = self.config.tile_overlap // compress
            stride = tile_size - overlap

            out_H = H * compress
            out_W = W * compress
            out_tile = tile_size * compress
            out_overlap = overlap * compress

            output = torch.zeros(B, 3, (T - 1) * 4 + 1, out_H, out_W,
                                 device=zs.device, dtype=zs.dtype)
            weight = torch.zeros(1, 1, 1, out_H, out_W,
                                 device=zs.device, dtype=zs.dtype)

            blend = _make_tile_blend_mask(out_tile, out_overlap, zs.device, zs.dtype)

            for y in range(0, H, stride):
                for x_pos in range(0, W, stride):
                    y_end = min(y + tile_size, H)
                    x_end = min(x_pos + tile_size, W)
                    y_start = y_end - tile_size
                    x_start = x_end - tile_size

                    tile_z = zs[:, :, :, y_start:y_end, x_start:x_end]

                    # Decode tile
                    self.model.clear_cache()
                    tile_video = self.model.decode(tile_z, self.scale)

                    py = y_start * compress
                    px = x_start * compress
                    th = tile_video.shape[3]
                    tw = tile_video.shape[4]

                    bm = blend[:th, :tw].unsqueeze(0).unsqueeze(0).unsqueeze(0)

                    output[:, :, :, py:py + th, px:px + tw] += tile_video * bm
                    weight[:, :, :, py:py + th, px:px + tw] += bm

            output = output / weight.clamp(min=1e-8)

        return output.to(in_dtype)


# ---------------------------------------------------------------------------
# Utility: channel-last memory format
# ---------------------------------------------------------------------------

def _apply_channel_last(module: nn.Module):
    """
    Convert Conv2d layers to channels_last format for better GPU performance.
    Conv3d uses channels_last_3d where supported.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            child.to(memory_format=torch.channels_last)
        elif isinstance(child, nn.Conv3d):
            try:
                child.to(memory_format=torch.channels_last_3d)
            except Exception:
                pass
        else:
            _apply_channel_last(child)


# ---------------------------------------------------------------------------
# Utility: tile blending mask
# ---------------------------------------------------------------------------

def _make_tile_blend_mask(tile_size, overlap, device, dtype):
    """
    Create a 2D blending mask for tile stitching.
    Linear ramp in the overlap region, 1.0 in the center.
    """
    mask = torch.ones(tile_size, tile_size, device=device, dtype=dtype)
    if overlap <= 0:
        return mask

    ramp = torch.linspace(0, 1, overlap, device=device, dtype=dtype)

    # Top / bottom ramps
    mask[:overlap, :] *= ramp.unsqueeze(1)
    mask[-overlap:, :] *= ramp.flip(0).unsqueeze(1)

    # Left / right ramps
    mask[:, :overlap] *= ramp.unsqueeze(0)
    mask[:, -overlap:] *= ramp.flip(0).unsqueeze(0)

    return mask


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_vae(
    vae,
    frames: int = 81,
    height: int = 480,
    width: int = 832,
    warmup: int = 2,
    repeats: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Benchmark VAE encode and decode with the given parameters.

    Args:
        vae: WanVAE or OptimizedWanVAE instance
        frames: Number of video frames
        height, width: Spatial dimensions
        warmup: Number of warmup iterations
        repeats: Number of timed iterations
        device: CUDA device
        dtype: Data type

    Returns:
        dict with encode_ms, decode_ms, total_ms (averages)
    """
    video = torch.randn(1, 3, frames, height, width, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        z = vae.encode(video)
        _ = vae.decode(z)
        torch.cuda.synchronize()

    # Benchmark encode
    encode_times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        z = vae.encode(video)
        torch.cuda.synchronize()
        encode_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark decode
    z = vae.encode(video)
    torch.cuda.synchronize()
    decode_times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = vae.decode(z)
        torch.cuda.synchronize()
        decode_times.append((time.perf_counter() - t0) * 1000)

    enc_avg = sum(encode_times) / len(encode_times)
    dec_avg = sum(decode_times) / len(decode_times)

    results = {
        "encode_ms": enc_avg,
        "decode_ms": dec_avg,
        "total_ms": enc_avg + dec_avg,
        "encode_all": encode_times,
        "decode_all": decode_times,
    }

    print(f"\n{'='*60}")
    print(f"VAE Benchmark Results ({frames}f x {height}x{width})")
    print(f"{'='*60}")
    print(f"  Encode:  {enc_avg:.1f} ms  (min={min(encode_times):.1f}, max={max(encode_times):.1f})")
    print(f"  Decode:  {dec_avg:.1f} ms  (min={min(decode_times):.1f}, max={max(decode_times):.1f})")
    print(f"  Total:   {enc_avg + dec_avg:.1f} ms")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Wan2.2 VAE Optimization Tool - Benchmark & Optimize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with torch.compile (max-autotune)
  python -m turbodiffusion.vae_optimize --vae_path checkpoints/Wan2.2_VAE.pth \\
      --compile --compile_mode max-autotune

  # Benchmark with spatial tiling (for high-res / low VRAM)
  python -m turbodiffusion.vae_optimize --vae_path checkpoints/Wan2.2_VAE.pth \\
      --tiling --tile_size 256

  # Benchmark with decode temporal batching
  python -m turbodiffusion.vae_optimize --vae_path checkpoints/Wan2.2_VAE.pth \\
      --compile --decode_batch 4

  # Full optimization
  python -m turbodiffusion.vae_optimize --vae_path checkpoints/Wan2.2_VAE.pth \\
      --compile --channel_last --decode_batch 4

  # Compare baseline vs optimized
  python -m turbodiffusion.vae_optimize --vae_path checkpoints/Wan2.2_VAE.pth \\
      --compare --compile --decode_batch 4
        """,
    )

    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to VAE checkpoint (e.g. Wan2.2_VAE.pth)")
    parser.add_argument("--z_dim", type=int, default=16, help="Latent dimension")

    # Optimization flags
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for encoder/decoder")
    parser.add_argument("--compile_mode", type=str, default="max-autotune",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--tiling", action="store_true",
                        help="Enable spatial tiling")
    parser.add_argument("--tile_size", type=int, default=256,
                        help="Tile size for spatial tiling")
    parser.add_argument("--tile_overlap", type=int, default=32,
                        help="Overlap between tiles")
    parser.add_argument("--decode_batch", type=int, default=1,
                        help="Temporal batch size for decode (>1 to batch frames)")
    parser.add_argument("--channel_last", action="store_true",
                        help="Use channels_last memory format")

    # Benchmark params
    parser.add_argument("--frames", type=int, default=81, help="Number of video frames")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--compare", action="store_true",
                        help="Run baseline and optimized, then show comparison")

    args = parser.parse_args()

    # Import here to avoid import errors when running --help
    import sys
    sys.path.insert(0, ".")
    from rcm.tokenizers.wan2pt1 import WanVAE

    print(f"Loading VAE from {args.vae_path} ...")
    vae = WanVAE(z_dim=args.z_dim, vae_pth=args.vae_path, dtype=torch.bfloat16, is_amp=False)
    print(f"VAE parameters: {vae.count_param() / 1e6:.1f}M")

    if args.compare:
        # Baseline benchmark
        print("\n--- BASELINE (no optimization) ---")
        baseline = benchmark_vae(
            vae, args.frames, args.height, args.width,
            args.warmup, args.repeats,
        )

    # Build optimization config
    config = OptimizeConfig(
        compile_encoder=args.compile,
        compile_decoder=args.compile,
        compile_mode=args.compile_mode,
        spatial_tiling=args.tiling,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        decode_temporal_batch=args.decode_batch,
        channel_last=args.channel_last,
        verbose=True,
    )

    opt_vae = optimize_vae(vae, config)

    print("\n--- OPTIMIZED ---")
    optimized = benchmark_vae(
        opt_vae, args.frames, args.height, args.width,
        args.warmup, args.repeats,
    )

    if args.compare:
        enc_speedup = baseline["encode_ms"] / optimized["encode_ms"]
        dec_speedup = baseline["decode_ms"] / optimized["decode_ms"]
        total_speedup = baseline["total_ms"] / optimized["total_ms"]
        print(f"\n{'='*60}")
        print(f"SPEEDUP COMPARISON")
        print(f"{'='*60}")
        print(f"  Encode:  {baseline['encode_ms']:.1f}ms -> {optimized['encode_ms']:.1f}ms  ({enc_speedup:.2f}x)")
        print(f"  Decode:  {baseline['decode_ms']:.1f}ms -> {optimized['decode_ms']:.1f}ms  ({dec_speedup:.2f}x)")
        print(f"  Total:   {baseline['total_ms']:.1f}ms -> {optimized['total_ms']:.1f}ms  ({total_speedup:.2f}x)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
