# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
AutoencoderKLWan3_8 → ONNX → TensorRT Export & Inference

Exports the VAE encoder and decoder as static-shape ONNX graphs with all
GPU-compatible ops, then builds TensorRT engines with BF16 precision.

The key insight: when called WITHOUT feat_cache (cache=None), the decoder
and encoder run a clean single-pass forward — no stateful caching needed.
This makes ONNX tracing straightforward.

Pipeline:
  1. Wrap encoder/decoder into traceable nn.Module (no cache, no loops)
  2. Export to ONNX with static shapes (opset 17, all ops GPU-safe)
  3. Build TensorRT engines with BF16 via trtexec or Python API
  4. TensorRTVAE wrapper provides encode()/decode() matching the
     original AutoencoderKLWan3_8 interface

Usage:
    from turbodiffusion.vae_tensorrt import (
        export_vae_onnx, build_tensorrt_engines, TensorRTVAE
    )

    # Step 1: Export ONNX
    export_vae_onnx(
        vae,                         # AutoencoderKLWan3_8 instance
        output_dir="./vae_onnx",
        latent_shape=(1, 48, 21, 60, 104),  # B,C,T,H,W latent
        video_shape=(1, 3, 81, 480, 832),   # B,C,T,H,W pixel
    )

    # Step 2: Build TensorRT engines
    build_tensorrt_engines(
        onnx_dir="./vae_onnx",
        engine_dir="./vae_trt",
        bf16=True,
    )

    # Step 3: Inference
    trt_vae = TensorRTVAE("./vae_trt", scale=vae.scale, z_dim=48)
    decoded = trt_vae.decode(latent_z)
    encoded = trt_vae.encode(pixel_video)
"""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patchify / Unpatchify (self-contained, no import dependency)
# ---------------------------------------------------------------------------

def _patchify(x, patch_size):
    if patch_size <= 1:
        return x
    if x.dim() == 5:
        return rearrange(
            x, "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size, r=patch_size,
        )
    raise ValueError(f"Expected 5D tensor, got {x.dim()}D")


def _unpatchify(x, patch_size):
    if patch_size <= 1:
        return x
    if x.dim() == 5:
        return rearrange(
            x, "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size, r=patch_size,
        )
    raise ValueError(f"Expected 5D tensor, got {x.dim()}D")


# ---------------------------------------------------------------------------
# ONNX-exportable wrappers (no caching, single-pass)
# ---------------------------------------------------------------------------

class _DecoderForExport(nn.Module):
    """
    Wraps the inner AutoencoderKLWan2_2_ decode path into a single-pass
    nn.Module suitable for ONNX tracing.

    Forward: latent z → undo_scale → conv2 → decoder → unpatchify → clamp → output

    All ops are static and GPU-friendly. No dynamic caching or Python loops
    over temporal frames — the full temporal dim is processed at once.
    """

    def __init__(self, inner_model, scale, patch_size=2):
        super().__init__()
        self.conv2 = inner_model.conv2
        self.decoder = inner_model.decoder
        self.z_dim = inner_model.z_dim
        self.patch_size = patch_size

        # Register scale as buffers so they travel with the model
        self.register_buffer("scale_mean", scale[0].clone())
        self.register_buffer("scale_inv_std", scale[1].clone())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Undo scale normalization: z_raw = z / inv_std + mean
        z = (z / self.scale_inv_std.view(1, self.z_dim, 1, 1, 1)
             + self.scale_mean.view(1, self.z_dim, 1, 1, 1))

        # Post-quant conv
        x = self.conv2(z)

        # Full decoder pass (no caching → processes all frames at once)
        out = self.decoder(x)

        # Unpatchify
        if self.patch_size > 1:
            out = rearrange(
                out,
                "b (c r q) f h w -> b c f (h q) (w r)",
                q=self.patch_size, r=self.patch_size,
            )

        return out.clamp(-1.0, 1.0)


class _EncoderForExport(nn.Module):
    """
    Wraps the inner AutoencoderKLWan2_2_ encode path into a single-pass
    nn.Module suitable for ONNX tracing.

    Forward: pixel video → patchify → encoder → conv1 → scale → [mu, logvar]
    """

    def __init__(self, inner_model, scale, patch_size=2):
        super().__init__()
        self.encoder = inner_model.encoder
        self.conv1 = inner_model.conv1
        self.z_dim = inner_model.z_dim
        self.patch_size = patch_size

        self.register_buffer("scale_mean", scale[0].clone())
        self.register_buffer("scale_inv_std", scale[1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patchify: [B, 3, T, H, W] → [B, 12, T, H/2, W/2]
        if self.patch_size > 1:
            x = rearrange(
                x,
                "b c f (h q) (w r) -> b (c r q) f h w",
                q=self.patch_size, r=self.patch_size,
            )

        # Full encoder pass (no caching)
        out = self.encoder(x)

        # Quant conv → split mu, logvar
        out = self.conv1(out)
        mu, log_var = out.chunk(2, dim=1)

        # Apply scale normalization
        mu = ((mu - self.scale_mean.view(1, self.z_dim, 1, 1, 1))
              * self.scale_inv_std.view(1, self.z_dim, 1, 1, 1))

        return torch.cat([mu, log_var], dim=1)


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

@dataclass
class OnnxExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 17
    # Latent shape: (batch, channels, temporal, height, width)
    # For 480p 81-frame video with 4x temporal + 8x spatial compression:
    #   latent = (1, 48, 21, 60, 104)
    latent_shape: Tuple[int, ...] = (1, 48, 21, 60, 104)
    # Video pixel shape: (batch, 3, frames, height, width)
    video_shape: Tuple[int, ...] = (1, 3, 81, 480, 832)
    # Export components
    export_decoder: bool = True
    export_encoder: bool = True
    # Patch size used by AutoencoderKLWan3_8
    patch_size: int = 2


def export_vae_onnx(
    vae,
    output_dir: str = "./vae_onnx",
    config: Optional[OnnxExportConfig] = None,
    device: str = "cuda",
):
    """
    Export AutoencoderKLWan3_8 encoder/decoder to static-shape ONNX.

    Args:
        vae: AutoencoderKLWan3_8 instance (with vae.model, vae.scale)
        output_dir: Directory for ONNX files
        config: Export configuration
        device: Export device
    """
    if config is None:
        config = OnnxExportConfig()

    os.makedirs(output_dir, exist_ok=True)

    # --- Detect inner model ---
    inner = getattr(vae, "model", None)
    if inner is None or not hasattr(inner, "clear_cache"):
        raise ValueError(
            f"Expected AutoencoderKLWan3_8 with vae.model (AutoencoderKLWan2_2_), "
            f"got {type(vae).__name__}"
        )
    scale = getattr(vae, "scale", None)
    if scale is None:
        raise ValueError("vae.scale not found — expected [mean, inv_std] tensors")

    # Convert scale to tensors
    scale_tensors = [
        s.clone().float() if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
        for s in scale
    ]

    logger.info(f"[TRT Export] Inner model: {type(inner).__name__}")
    logger.info(f"[TRT Export] z_dim={inner.z_dim}, patch_size={config.patch_size}")
    logger.info(f"[TRT Export] Latent shape: {config.latent_shape}")
    logger.info(f"[TRT Export] Video shape:  {config.video_shape}")

    # --- Export decoder ---
    if config.export_decoder:
        logger.info("[TRT Export] Exporting decoder to ONNX ...")
        dec_wrapper = _DecoderForExport(
            inner, scale_tensors, patch_size=config.patch_size
        ).to(device).eval()

        dummy_z = torch.randn(config.latent_shape, dtype=torch.float32, device=device)
        dec_onnx_path = os.path.join(output_dir, "vae_decoder.onnx")

        with torch.no_grad():
            torch.onnx.export(
                dec_wrapper,
                (dummy_z,),
                dec_onnx_path,
                input_names=["latent_z"],
                output_names=["decoded_video"],
                opset_version=config.opset_version,
                do_constant_folding=True,
                dynamo=False,
            )

        logger.info(f"[TRT Export] Decoder ONNX saved: {dec_onnx_path}")
        _verify_onnx(dec_onnx_path)

    # --- Export encoder ---
    if config.export_encoder:
        logger.info("[TRT Export] Exporting encoder to ONNX ...")
        enc_wrapper = _EncoderForExport(
            inner, scale_tensors, patch_size=config.patch_size
        ).to(device).eval()

        dummy_video = torch.randn(config.video_shape, dtype=torch.float32, device=device)
        enc_onnx_path = os.path.join(output_dir, "vae_encoder.onnx")

        with torch.no_grad():
            torch.onnx.export(
                enc_wrapper,
                (dummy_video,),
                enc_onnx_path,
                input_names=["pixel_video"],
                output_names=["latent_dist"],
                opset_version=config.opset_version,
                do_constant_folding=True,
                dynamo=False,
            )

        logger.info(f"[TRT Export] Encoder ONNX saved: {enc_onnx_path}")
        _verify_onnx(enc_onnx_path)

    logger.info(f"[TRT Export] ONNX export complete → {output_dir}")


def _verify_onnx(onnx_path: str):
    """Verify the exported ONNX model is valid."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model, full_check=True)
        logger.info(f"[TRT Export] ONNX verification passed: {onnx_path}")

        # Log graph stats
        graph = model.graph
        logger.info(
            f"[TRT Export]   nodes={len(graph.node)}, "
            f"inputs={[f'{i.name}: {[d.dim_value for d in i.type.tensor_type.shape.dim]}' for i in graph.input]}, "
            f"outputs={[o.name for o in graph.output]}"
        )
    except ImportError:
        logger.warning("[TRT Export] onnx package not installed, skipping verification")
    except Exception as e:
        logger.warning(f"[TRT Export] ONNX verification warning: {e}")


# ---------------------------------------------------------------------------
# TensorRT Engine Build
# ---------------------------------------------------------------------------

@dataclass
class TrtBuildConfig:
    """Configuration for TensorRT engine build."""
    bf16: bool = True
    fp16: bool = False  # fallback if BF16 not supported
    max_workspace_gb: int = 8
    # Builder optimization level (0-5, higher = slower build, faster inference)
    builder_optimization_level: int = 3
    # Timing cache for faster rebuilds
    timing_cache_path: Optional[str] = None
    # Use trtexec CLI instead of Python API
    use_trtexec: bool = False


def build_tensorrt_engines(
    onnx_dir: str = "./vae_onnx",
    engine_dir: str = "./vae_trt",
    config: Optional[TrtBuildConfig] = None,
):
    """
    Build TensorRT engines from exported ONNX models.

    Args:
        onnx_dir: Directory containing vae_decoder.onnx / vae_encoder.onnx
        engine_dir: Output directory for .engine files
        config: TensorRT build configuration
    """
    if config is None:
        config = TrtBuildConfig()

    os.makedirs(engine_dir, exist_ok=True)

    onnx_files = {
        "decoder": os.path.join(onnx_dir, "vae_decoder.onnx"),
        "encoder": os.path.join(onnx_dir, "vae_encoder.onnx"),
    }

    for name, onnx_path in onnx_files.items():
        if not os.path.exists(onnx_path):
            logger.info(f"[TRT Build] Skipping {name} (ONNX not found: {onnx_path})")
            continue

        engine_path = os.path.join(engine_dir, f"vae_{name}.engine")

        if config.use_trtexec:
            _build_with_trtexec(onnx_path, engine_path, config)
        else:
            _build_with_python_api(onnx_path, engine_path, config)


def _build_with_trtexec(onnx_path: str, engine_path: str, config: TrtBuildConfig):
    """Build TensorRT engine using trtexec CLI."""
    import subprocess

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={config.max_workspace_gb * 1024}",
        f"--builderOptimizationLevel={config.builder_optimization_level}",
    ]

    if config.bf16:
        cmd.append("--bf16")
    elif config.fp16:
        cmd.append("--fp16")

    if config.timing_cache_path:
        cmd.append(f"--timingCacheFile={config.timing_cache_path}")

    # Enable all GPU-safe tactics
    cmd.extend([
        "--noTF32",          # Disable TF32 when using BF16 for consistency
        "--verbose",
    ])

    logger.info(f"[TRT Build] Running: {' '.join(cmd)}")
    t0 = time.perf_counter()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"[TRT Build] trtexec failed:\n{result.stderr}")
        raise RuntimeError(f"trtexec failed for {onnx_path}")

    elapsed = time.perf_counter() - t0
    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info(
        f"[TRT Build] Engine built: {engine_path} "
        f"({engine_size_mb:.1f} MB, {elapsed:.1f}s)"
    )


def _build_with_python_api(onnx_path: str, engine_path: str, config: TrtBuildConfig):
    """Build TensorRT engine using the Python API."""
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "tensorrt package not found. Install with:\n"
            "  pip install tensorrt tensorrt-cu12\n"
            "Or use config.use_trtexec=True to build via trtexec CLI."
        )

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    logger.info(f"[TRT Build] Building engine for: {onnx_path}")
    logger.info(f"[TRT Build] TensorRT version: {trt.__version__}")
    t0 = time.perf_counter()

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"[TRT Build] ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    logger.info(
        f"[TRT Build] Network: {network.num_inputs} inputs, "
        f"{network.num_outputs} outputs, {network.num_layers} layers"
    )

    # Log input/output shapes
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        logger.info(f"[TRT Build]   Input  {i}: {inp.name} {inp.shape} {inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        logger.info(f"[TRT Build]   Output {i}: {out.name} {out.shape} {out.dtype}")

    # Builder config
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        config.max_workspace_gb * (1 << 30),
    )
    builder_config.builder_optimization_level = config.builder_optimization_level

    # --- BF16 precision ---
    if config.bf16:
        if builder.platform_has_fast_bf16:
            builder_config.set_flag(trt.BuilderFlag.BF16)
            logger.info("[TRT Build] BF16 precision enabled")
        else:
            logger.warning(
                "[TRT Build] BF16 not supported on this GPU, falling back to FP16"
            )
            if builder.platform_has_fast_fp16:
                builder_config.set_flag(trt.BuilderFlag.FP16)
    elif config.fp16:
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            logger.info("[TRT Build] FP16 precision enabled")

    # Timing cache
    timing_cache_data = b""
    if config.timing_cache_path and os.path.exists(config.timing_cache_path):
        with open(config.timing_cache_path, "rb") as f:
            timing_cache_data = f.read()
        logger.info(f"[TRT Build] Loaded timing cache: {config.timing_cache_path}")

    timing_cache = builder_config.create_timing_cache(timing_cache_data)
    builder_config.set_timing_cache(timing_cache, ignore_mismatch=False)

    # Build engine
    logger.info("[TRT Build] Building engine (this may take several minutes) ...")
    engine_bytes = builder.build_serialized_network(network, builder_config)

    if engine_bytes is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    # Save timing cache
    if config.timing_cache_path:
        updated_cache = builder_config.get_timing_cache()
        with open(config.timing_cache_path, "wb") as f:
            f.write(bytearray(updated_cache.serialize()))
        logger.info(f"[TRT Build] Updated timing cache: {config.timing_cache_path}")

    elapsed = time.perf_counter() - t0
    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info(
        f"[TRT Build] Engine built: {engine_path} "
        f"({engine_size_mb:.1f} MB, {elapsed:.1f}s)"
    )


# ---------------------------------------------------------------------------
# TensorRT Inference Wrapper
# ---------------------------------------------------------------------------

class TensorRTVAE:
    """
    Drop-in VAE replacement using TensorRT engines.

    Provides encode() and decode() matching the AutoencoderKLWan3_8 interface.

    Usage:
        trt_vae = TensorRTVAE(
            engine_dir="./vae_trt",
            scale=original_vae.scale,
            z_dim=48,
        )
        # Decode
        decoded = trt_vae.decode(latent_z)  # [B,48,T,H,W] → [B,3,T*4,H*8,W*8]
        # Encode
        encoded = trt_vae.encode(video)     # [B,3,T,H,W] → DiagonalGaussianDistribution
    """

    def __init__(
        self,
        engine_dir: str,
        scale: Optional[list] = None,
        z_dim: int = 48,
        device: str = "cuda",
    ):
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("tensorrt package required for TensorRTVAE inference")

        self.device = torch.device(device)
        self.z_dim = z_dim
        self.scale = scale
        self._trt = trt

        # Load engines
        self._decoder_engine = None
        self._decoder_context = None
        self._encoder_engine = None
        self._encoder_context = None

        dec_path = os.path.join(engine_dir, "vae_decoder.engine")
        enc_path = os.path.join(engine_dir, "vae_encoder.engine")

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        if os.path.exists(dec_path):
            self._decoder_engine = self._load_engine(dec_path)
            self._decoder_context = self._decoder_engine.create_execution_context()
            logger.info(f"[TRT VAE] Loaded decoder engine: {dec_path}")

        if os.path.exists(enc_path):
            self._encoder_engine = self._load_engine(enc_path)
            self._encoder_context = self._encoder_engine.create_execution_context()
            logger.info(f"[TRT VAE] Loaded encoder engine: {enc_path}")

        # Pre-allocate I/O buffers (will be resized on first call)
        self._buffers = {}

    def _load_engine(self, engine_path: str):
        with open(engine_path, "rb") as f:
            return self._runtime.deserialize_cuda_engine(f.read())

    def _ensure_buffers(self, context, engine, inputs: dict):
        """Allocate output buffers and set tensor addresses."""
        buffers = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)

            if mode == self._trt.TensorIOMode.INPUT:
                # Use provided input tensor
                tensor = inputs[name].contiguous()
                context.set_input_shape(name, tuple(tensor.shape))
                context.set_tensor_address(name, tensor.data_ptr())
                buffers[name] = tensor
            else:
                # Allocate output buffer
                shape = context.get_tensor_shape(name)
                dtype = self._trt_dtype_to_torch(engine.get_tensor_dtype(name))
                tensor = torch.empty(
                    tuple(shape), dtype=dtype, device=self.device
                )
                context.set_tensor_address(name, tensor.data_ptr())
                buffers[name] = tensor

        return buffers

    def _trt_dtype_to_torch(self, trt_dtype):
        mapping = {
            self._trt.DataType.FLOAT: torch.float32,
            self._trt.DataType.HALF: torch.float16,
            self._trt.DataType.BF16: torch.bfloat16,
            self._trt.DataType.INT8: torch.int8,
            self._trt.DataType.INT32: torch.int32,
        }
        return mapping.get(trt_dtype, torch.float32)

    @torch.no_grad()
    def decode(self, z: torch.Tensor, return_dict: bool = True):
        """
        Decode latent z to pixel video using TensorRT.

        Args:
            z: Latent tensor [B, C, T, H, W] (already scaled)
            return_dict: If True, return DecoderOutput

        Returns:
            Decoded video [B, 3, T_out, H_out, W_out] clamped to [-1, 1]
        """
        if self._decoder_context is None:
            raise RuntimeError("Decoder engine not loaded")

        z_input = z.to(dtype=torch.float32, device=self.device).contiguous()

        # Set up I/O
        buffers = self._ensure_buffers(
            self._decoder_context, self._decoder_engine,
            {"latent_z": z_input},
        )

        # Execute
        stream = torch.cuda.current_stream()
        self._decoder_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        decoded = buffers["decoded_video"]

        if not return_dict:
            return (decoded,)
        from diffusers.models.autoencoders.vae import DecoderOutput
        return DecoderOutput(sample=decoded)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        """
        Encode pixel video to latent distribution using TensorRT.

        Args:
            x: Pixel video [B, 3, T, H, W]
            return_dict: If True, return AutoencoderKLOutput

        Returns:
            Latent distribution (DiagonalGaussianDistribution)
        """
        if self._encoder_context is None:
            raise RuntimeError("Encoder engine not loaded")

        x_input = x.to(dtype=torch.float32, device=self.device).contiguous()

        buffers = self._ensure_buffers(
            self._encoder_context, self._encoder_engine,
            {"pixel_video": x_input},
        )

        stream = torch.cuda.current_stream()
        self._encoder_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        h = buffers["latent_dist"]

        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        from diffusers.models.modeling_outputs import AutoencoderKLOutput
        return AutoencoderKLOutput(latent_dist=posterior)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="AutoencoderKLWan3_8 → ONNX → TensorRT pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- export-onnx ---
    p_export = sub.add_parser("export-onnx", help="Export VAE to ONNX")
    p_export.add_argument("--vae_path", required=True, help="Path to VAE checkpoint")
    p_export.add_argument("--output_dir", default="./vae_onnx")
    p_export.add_argument(
        "--latent_shape", type=int, nargs=5, default=[1, 48, 21, 60, 104],
        help="B C T H W for latent input",
    )
    p_export.add_argument(
        "--video_shape", type=int, nargs=5, default=[1, 3, 81, 480, 832],
        help="B C T H W for video input",
    )
    p_export.add_argument("--decoder_only", action="store_true")
    p_export.add_argument("--encoder_only", action="store_true")
    p_export.add_argument("--opset", type=int, default=17)
    p_export.add_argument("--device", default="cuda")

    # --- build-trt ---
    p_build = sub.add_parser("build-trt", help="Build TensorRT engines from ONNX")
    p_build.add_argument("--onnx_dir", default="./vae_onnx")
    p_build.add_argument("--engine_dir", default="./vae_trt")
    p_build.add_argument("--bf16", action="store_true", default=True)
    p_build.add_argument("--fp16", action="store_true")
    p_build.add_argument("--workspace_gb", type=int, default=8)
    p_build.add_argument("--opt_level", type=int, default=3)
    p_build.add_argument("--timing_cache", default=None)
    p_build.add_argument("--trtexec", action="store_true",
                         help="Use trtexec CLI instead of Python API")

    # --- full-pipeline ---
    p_full = sub.add_parser("full", help="Export ONNX + Build TensorRT in one step")
    p_full.add_argument("--vae_path", required=True)
    p_full.add_argument("--onnx_dir", default="./vae_onnx")
    p_full.add_argument("--engine_dir", default="./vae_trt")
    p_full.add_argument(
        "--latent_shape", type=int, nargs=5, default=[1, 48, 21, 60, 104],
    )
    p_full.add_argument(
        "--video_shape", type=int, nargs=5, default=[1, 3, 81, 480, 832],
    )
    p_full.add_argument("--bf16", action="store_true", default=True)
    p_full.add_argument("--workspace_gb", type=int, default=8)
    p_full.add_argument("--device", default="cuda")
    p_full.add_argument("--trtexec", action="store_true")

    args = parser.parse_args()

    if args.command in ("export-onnx", "full"):
        # Load VAE
        logger.info(f"Loading VAE from {args.vae_path} ...")
        vae = _load_vae_for_export(args.vae_path, args.device)

        onnx_cfg = OnnxExportConfig(
            latent_shape=tuple(args.latent_shape),
            video_shape=tuple(args.video_shape),
            opset_version=getattr(args, "opset", 17),
            export_decoder=not getattr(args, "encoder_only", False),
            export_encoder=not getattr(args, "decoder_only", False),
        )
        output_dir = getattr(args, "output_dir", args.onnx_dir)
        export_vae_onnx(vae, output_dir=output_dir, config=onnx_cfg, device=args.device)

    if args.command in ("build-trt", "full"):
        onnx_dir = args.onnx_dir
        trt_cfg = TrtBuildConfig(
            bf16=args.bf16,
            fp16=getattr(args, "fp16", False),
            max_workspace_gb=args.workspace_gb,
            builder_optimization_level=getattr(args, "opt_level", 3),
            timing_cache_path=getattr(args, "timing_cache", None),
            use_trtexec=getattr(args, "trtexec", False),
        )
        build_tensorrt_engines(
            onnx_dir=onnx_dir,
            engine_dir=args.engine_dir,
            config=trt_cfg,
        )

    logger.info("Done!")


def _load_vae_for_export(vae_path: str, device: str = "cuda"):
    """
    Load a VAE checkpoint for export.
    Tries AutoencoderKLWan3_8 first, then safetensors/torch state dict.
    """
    # Try importing from the file that defines AutoencoderKLWan3_8
    # The user's custom module path may vary
    try:
        from wan.modules.vae import AutoencoderKLWan3_8
        vae = AutoencoderKLWan3_8.from_pretrained(vae_path)
        vae = vae.to(device).eval()
        return vae
    except ImportError:
        pass

    # Try diffusers
    try:
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch.float32)
        vae = vae.to(device).eval()
        return vae
    except Exception:
        pass

    raise RuntimeError(
        f"Could not load VAE from {vae_path}. "
        f"Provide an AutoencoderKLWan3_8 checkpoint or diffusers model path."
    )


if __name__ == "__main__":
    main()
