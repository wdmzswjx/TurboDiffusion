"""
Inference script for Wan2.2-Fun-5B with INT8 quantization and VRAM optimization.

Supports:
- INT8 quantized linear layers (via CUTLASS INT8 GEMM)
- Fast Triton-based RMSNorm / LayerNorm
- SLA / SageSLA sparse attention
- Dual-model high/low noise with CPU offloading for VRAM savings
- Adaptive resolution based on input image aspect ratio

Usage:
    python wan2.2_fun_infer.py \
        --image_path input.jpg \
        --high_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-high-quant.pth \
        --low_noise_model_path checkpoints/modified/Wan2.2-Fun-5B-low-quant.pth \
        --prompt "A cat walking on the beach" \
        --quant_linear \
        --num_steps 4
"""

import argparse
import math

import torch
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TurboDiffusion inference for Wan2.2-Fun-5B with quantization & VRAM optimization"
    )
    # Model paths
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image")
    parser.add_argument("--high_noise_model_path", type=str, required=True, help="Path to the high-noise model")
    parser.add_argument("--low_noise_model_path", type=str, required=True, help="Path to the low-noise model")
    parser.add_argument("--boundary", type=float, default=0.9, help="Timestep boundary for switching models")
    parser.add_argument("--model", choices=["Wan2.2-Fun-5B"], default="Wan2.2-Fun-5B", help="Model variant")

    # Sampling
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 steps for distilled inference")
    parser.add_argument("--sigma_max", type=float, default=200, help="Initial sigma for rCM")
    parser.add_argument("--ode", action="store_true", help="Use ODE sampling (sharper but less robust)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Paths
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth", help="Path to VAE")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to umT5")
    parser.add_argument("--save_path", type=str, default="output/wan22_fun_generated.mp4", help="Output path")

    # Resolution
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--resolution", default="720p", type=str, help="Resolution")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio (width:height)")
    parser.add_argument("--adaptive_resolution", action="store_true", help="Adapt resolution to input image")

    # Quantization & acceleration
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla", help="Attention type")
    parser.add_argument("--sla_topk", type=float, default=0.1, help="Top-k ratio for SLA/SageSLA")
    parser.add_argument("--quant_linear", action="store_true", help="Use INT8 quantized linear layers")
    parser.add_argument("--default_norm", action="store_true", help="Keep default norms")

    # VRAM optimization
    parser.add_argument("--offload_to_cpu", action="store_true", default=True,
                        help="Offload inactive model to CPU (enabled by default for dual-model)")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Pin CPU tensors for faster CPU-GPU transfer")

    # Server mode
    parser.add_argument("--serve", action="store_true", help="Launch interactive TUI server mode")
    return parser.parse_args()


def load_model_to_device(model, device, pin_memory=False):
    """Move model to target device with optional memory pinning."""
    if device == "cpu" and pin_memory:
        model = model.cpu()
        for param in model.parameters():
            param.data = param.data.pin_memory()
        for buf in model.buffers():
            if buf.is_floating_point() or buf.dtype == torch.int8:
                buf.data = buf.data.pin_memory()
    else:
        model = model.to(device)
    return model


def log_vram_usage(tag: str):
    """Log current GPU VRAM usage."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    log.info(f"[VRAM {tag}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


if __name__ == "__main__":
    args = parse_arguments()

    # Handle serve mode
    if args.serve:
        args.mode = "i2v"
        from serve.tui import main as serve_main
        serve_main(args)
        exit(0)

    # Validate
    if args.prompt is None:
        log.error("--prompt is required (unless using --serve mode)")
        exit(1)
    if args.image_path is None:
        log.error("--image_path is required (unless using --serve mode)")
        exit(1)

    # Step 1: Text embedding (load encoder, compute, then free)
    log.info(f"Computing embedding for prompt: {args.prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(**tensor_kwargs)
    clear_umt5_memory()
    log_vram_usage("after text encoder")

    # Step 2: Load models with CPU offloading
    log.info("Loading DiT models (Wan2.2-Fun-5B)...")
    high_noise_model = create_model(dit_path=args.high_noise_model_path, args=args)
    high_noise_model = load_model_to_device(high_noise_model, "cpu", pin_memory=args.pin_memory)
    torch.cuda.empty_cache()

    low_noise_model = create_model(dit_path=args.low_noise_model_path, args=args)
    low_noise_model = load_model_to_device(low_noise_model, "cpu", pin_memory=args.pin_memory)
    torch.cuda.empty_cache()
    log.success("Successfully loaded DiT models.")
    log_vram_usage("after model loading")

    # Step 3: VAE & image preprocessing
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    log.info(f"Loading and preprocessing image from: {args.image_path}")
    input_image = Image.open(args.image_path).convert("RGB")

    if args.adaptive_resolution:
        log.info("Adaptive resolution mode enabled.")
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w

        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)

        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride
        log.info(f"Adaptive resolution: {w}x{h}")
    else:
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        log.info(f"Fixed resolution: {w}x{h}")

    F = args.num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    image_transforms = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = image_transforms(input_image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)

    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)], dim=2
        )
        encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode
        torch.cuda.empty_cache()

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {
        "crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples),
        "y_B_C_T_H_W": y,
    }

    # Step 4: Sampling with dual-model CPU offloading
    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples, *state_shape,
        dtype=torch.float32, device=tensor_kwargs["device"], generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][:args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64, device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1

    # Start with high-noise model on GPU
    high_noise_model.cuda()
    net = high_noise_model
    switched = False
    log_vram_usage("before sampling")

    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
        # Switch from high-noise to low-noise model at boundary
        if t_cur.item() < args.boundary and not switched:
            if args.offload_to_cpu:
                high_noise_model.cpu()
                torch.cuda.empty_cache()
                log_vram_usage("after offloading high-noise model")

            low_noise_model.cuda()
            net = low_noise_model
            switched = True
            log.info("Switched to low noise model.")
            log_vram_usage("after loading low-noise model")

        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition,
            ).to(torch.float64)

            if args.ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape, dtype=torch.float32,
                    device=tensor_kwargs["device"], generator=generator,
                )

    samples = x.float()

    # Offload model, free VRAM for decoding
    if switched:
        low_noise_model.cpu()
    else:
        high_noise_model.cpu()
    torch.cuda.empty_cache()
    log_vram_usage("before VAE decode")

    # Step 5: Decode and save
    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = [video.float().cpu()]
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
    log.success(f"Video saved to {args.save_path}")
