#!/bin/bash
# =================================================================
# Data preparation script for Wan2.2-Fun-5B I2V training
#
# Converts raw video + text data into WebDataset tar shards.
#
# Expected input structure:
#   raw_data/
#   ├── video_0001.mp4
#   ├── video_0001.txt     (text prompt)
#   ├── video_0001.jpg     (first frame image, optional)
#   ├── video_0002.mp4
#   ├── video_0002.txt
#   ...
#
# Output:
#   assets/datasets/Wan2.2_Fun_5B_720p_I2V/
#   ├── shard_000000.tar
#   ├── shard_000001.tar
#   ...
#
# Each tar contains per sample:
#   latent.pt          - VAE-encoded video latent
#   embed.pt           - T5 text embedding
#   prompt.txt         - Raw text prompt
#   image_latent.pt    - VAE-encoded first frame
# =================================================================

export PYTHONPATH=turbodiffusion

INPUT_DIR=${1:-"raw_data"}
OUTPUT_DIR=${2:-"assets/datasets/Wan2.2_Fun_5B_720p_I2V"}
VAE_PATH=${3:-"assets/checkpoints/Wan2.1_VAE.pth"}
TEXT_ENCODER_PATH=${4:-"assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth"}
RESOLUTION=${5:-"720p"}
SHARD_SIZE=${6:-1000}  # samples per shard

echo "==================================================================="
echo "  Wan2.2-Fun-5B I2V Data Preparation"
echo "==================================================================="
echo "  Input:          $INPUT_DIR"
echo "  Output:         $OUTPUT_DIR"
echo "  VAE:            $VAE_PATH"
echo "  Text Encoder:   $TEXT_ENCODER_PATH"
echo "  Resolution:     $RESOLUTION"
echo "  Shard size:     $SHARD_SIZE"
echo "==================================================================="

python -c "
import os
import sys
import glob
import math
import tarfile
import io
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T

# Resolution map
res_map = {
    '480p': (480, 832),
    '720p': (720, 1280),
}

input_dir = '${INPUT_DIR}'
output_dir = '${OUTPUT_DIR}'
vae_path = '${VAE_PATH}'
text_encoder_path = '${TEXT_ENCODER_PATH}'
resolution = '${RESOLUTION}'
shard_size = ${SHARD_SIZE}

h, w = res_map[resolution]
os.makedirs(output_dir, exist_ok=True)

# Find all video files
videos = sorted(glob.glob(os.path.join(input_dir, '*.mp4')))
print(f'Found {len(videos)} videos')

if len(videos) == 0:
    print('No videos found. Please check your input directory.')
    sys.exit(1)

# Load VAE
print('Loading VAE...')
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
tokenizer = Wan2pt1VAEInterface(vae_pth=vae_path)

# Load text encoder
print('Loading text encoder...')
from rcm.utils.umt5 import get_umt5_embedding, clear_umt5_memory

# Process and write shards
num_shards = math.ceil(len(videos) / shard_size)
sample_idx = 0

for shard_idx in range(num_shards):
    shard_path = os.path.join(output_dir, f'shard_{shard_idx:06d}.tar')
    start = shard_idx * shard_size
    end = min(start + shard_size, len(videos))
    shard_videos = videos[start:end]

    with tarfile.open(shard_path, 'w') as tar:
        for video_path in tqdm(shard_videos, desc=f'Shard {shard_idx}/{num_shards}'):
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            prompt_path = os.path.splitext(video_path)[0] + '.txt'

            if not os.path.exists(prompt_path):
                print(f'Warning: No prompt for {video_path}, skipping')
                continue

            with open(prompt_path) as f:
                prompt = f.read().strip()

            # Get text embedding
            with torch.no_grad():
                text_emb = get_umt5_embedding(
                    checkpoint_path=text_encoder_path,
                    prompts=prompt
                ).cpu()

            # Load and process video
            import torchvision
            video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            # video: [T, H, W, C] uint8
            video = video.permute(3, 0, 1, 2).unsqueeze(0).float()  # [1, C, T, H, W]
            video = torch.nn.functional.interpolate(
                video.flatten(0, 1), size=(h, w), mode='bilinear', align_corners=False
            ).unflatten(0, (1, video.shape[1], video.shape[2]))
            video = video[:, :, :, :, :] / 127.5 - 1.0  # normalize to [-1, 1]

            with torch.no_grad():
                latent = tokenizer.encode(video.cuda()).cpu()
                # First frame image latent
                first_frame = video[:, :, :1, :, :]
                image_latent = tokenizer.encode(first_frame.cuda()).cpu()

            # Write to tar
            key = f'{sample_idx:08d}'

            # latent.pt
            buf = io.BytesIO()
            torch.save(latent.squeeze(0), buf)
            info = tarfile.TarInfo(name=f'{key}.latent.pt')
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)

            # embed.pt
            buf = io.BytesIO()
            torch.save(text_emb.squeeze(0), buf)
            info = tarfile.TarInfo(name=f'{key}.embed.pt')
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)

            # prompt.txt
            prompt_bytes = prompt.encode('utf-8')
            info = tarfile.TarInfo(name=f'{key}.prompt.txt')
            info.size = len(prompt_bytes)
            tar.addfile(info, io.BytesIO(prompt_bytes))

            # image_latent.pt
            buf = io.BytesIO()
            torch.save(image_latent.squeeze(0), buf)
            info = tarfile.TarInfo(name=f'{key}.image_latent.pt')
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)

            sample_idx += 1

    print(f'Written shard {shard_path} ({len(shard_videos)} samples)')

clear_umt5_memory()
print(f'Done! Total samples: {sample_idx}, Shards: {num_shards}')
print(f'Output directory: {output_dir}')
"
