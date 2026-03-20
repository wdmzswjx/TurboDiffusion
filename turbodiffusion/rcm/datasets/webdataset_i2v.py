"""
WebDataset loader for I2V (Image-to-Video) training.

Extends the standard T2V webdataset with support for image conditioning.

Expected tar shard contents per sample:
    - latent.pt: Pre-encoded video latent [C, T, H, W]
    - embed.pt: T5 text embedding [L, D]
    - prompt.txt: Text prompt
    - image_latent.pt: Pre-encoded first-frame image latent [C, 1, H, W]

If image_latent.pt is not present, the first frame of the video latent
is extracted automatically during training.
"""

import glob
import webdataset as wds
import torch
from torch.utils.data import DataLoader


def dict_collation_fn(samples):
    if not samples:
        return {}

    keys = samples[0].keys()
    batched_dict = {key: [] for key in keys}

    for sample in samples:
        for key in keys:
            batched_dict[key].append(sample[key])

    for key in keys:
        if isinstance(batched_dict[key][0], torch.Tensor):
            batched_dict[key] = torch.stack(batched_dict[key])

    return batched_dict


def create_dataloader_i2v(
    tar_path_pattern,
    batch_size,
    num_workers=8,
    shuffle_buffer=1000,
    prefetch_factor=2,
):
    """
    Create a WebDataset dataloader for I2V training.

    The rename step maps:
        latent.pt       -> latents
        embed.pt        -> t5_text_embeddings
        prompt.txt      -> prompts
        image_latent.pt -> image_latents (optional)
    """
    shards = glob.glob(tar_path_pattern)
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")

    # Check if image_latent.pt exists in shards
    # Use flexible rename that doesn't fail on missing keys
    rename_map = {
        "latents": "latent.pt",
        "t5_text_embeddings": "embed.pt",
        "prompts": "prompt.txt",
    }

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.shuffle(1000),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(shuffle_buffer),
        wds.decode(wds.handle_extension("pt", wds.torch_loads)),
        # Rename standard keys
        wds.rename(**rename_map),
        # Add image_latent if present (via map)
        wds.map(_extract_image_latent),
        wds.batched(batch_size, partial=False, collation_fn=dict_collation_fn),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return dataloader


def _extract_image_latent(sample):
    """
    Post-processing step to handle image latent.

    If 'image_latent.pt' is present in the sample, rename it.
    Otherwise, extract from the first temporal frame of the video latent.
    """
    # Check if image_latent was decoded from tar
    if "image_latent.pt" in sample:
        sample["image_latents"] = sample.pop("image_latent.pt")
    elif "latents" in sample:
        # Extract first frame from video latent as image conditioning
        latent = sample["latents"]
        if latent.ndim == 4:  # [C, T, H, W]
            sample["image_latents"] = latent[:, :1, :, :]  # [C, 1, H, W]
        elif latent.ndim == 3:  # [C, H, W] - single frame
            sample["image_latents"] = latent.unsqueeze(1)  # [C, 1, H, W]

    return sample
