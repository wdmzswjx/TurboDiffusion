# SPDX-License-Identifier: Apache-2.0
#
# I2V (Image-to-Video) distillation model for Wan2.2-Fun-5B based on rCM.
# Extends the T2V rCM distillation model with image conditioning support.

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Literal

import attrs
import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor

from imaginaire.lazy_config import LazyCall as L, LazyDict
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.utils import log, misc

from rcm.conditioner import DataType, TextCondition
from rcm.configs.defaults.ema import EMAConfig
from rcm.models.t2v_model_distill_rcm import (
    T2VDistillConfig_rCM,
    T2VDistillModel_rCM,
    DenoisePrediction,
    IS_PREPROCESSED_KEY,
    IS_PROCESSED_KEY,
)
from rcm.utils.timestep_utils import LogNormal


@attrs.define(slots=False)
class I2VDistillConfig_rCM(T2VDistillConfig_rCM):
    """Extends T2V config with I2V-specific parameters."""

    # I2V-specific
    input_image_key: str = "images"           # key in data_batch for input image
    input_image_latent_key: str = "image_latents"  # pre-encoded image latents
    state_ch: int = 16                        # output latent channels
    in_ch_with_image: int = 36                # 16 (latent) + 4 (mask) + 16 (image latent)
    mask_first_frame: bool = True             # apply mask to first frame for I2V conditioning


class I2VDistillModel_rCM(T2VDistillModel_rCM):
    """
    Image-to-Video distillation model using rCM (Rectified Consistency Model).

    Extends the T2V model with:
    1. Image encoding: first frame is encoded and concatenated as conditioning
    2. Mask construction: binary mask indicating which frames are conditioned
    3. y_B_C_T_H_W: concatenation of [mask, image_latents] passed to network

    The network architecture (WanModel with model_type="i2v") expects:
    - x_B_C_T_H_W: noisy latent [B, 16, T, H, W]
    - y_B_C_T_H_W: image conditioning [B, 20, T, H, W] = [4 mask + 16 image_latent]
    The model internally concatenates these to get [B, 36, T, H, W] input.
    """

    def __init__(self, config: I2VDistillConfig_rCM):
        super().__init__(config)

    def _build_image_condition(
        self,
        image_latents: torch.Tensor,
        state_shape: tuple,
    ) -> torch.Tensor:
        """
        Build the I2V conditioning tensor y_B_C_T_H_W.

        Args:
            image_latents: [B, C_lat, 1, H_lat, W_lat] or [B, C_lat, T, H, W]
                           Encoded first-frame image latents.
            state_shape: (C, T, H, W) of the target latent space.

        Returns:
            y_B_C_T_H_W: [B, 4+C_lat, T, H, W] conditioning tensor.
        """
        B = image_latents.shape[0]
        C_lat, T, H, W = state_shape

        # Mask: 1 for conditioned frames, 0 for unconditioned
        msk = torch.zeros(B, 4, T, H, W, device=image_latents.device, dtype=image_latents.dtype)
        if self.config.mask_first_frame:
            msk[:, :, 0, :, :] = 1.0

        # Image latents: place encoded image at first frame, zeros elsewhere
        img_lat = torch.zeros(B, C_lat, T, H, W, device=image_latents.device, dtype=image_latents.dtype)
        if image_latents.shape[2] == 1:
            img_lat[:, :, 0, :, :] = image_latents[:, :, 0, :, :]
        else:
            img_lat = image_latents

        # Concatenate: [mask(4), image_latent(C_lat)] -> [B, 4+C_lat, T, H, W]
        y = torch.cat([msk, img_lat], dim=1)
        return y

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor, TextCondition, TextCondition]:
        """
        Process data batch for I2V training.

        Expected data_batch keys:
        - "videos" or "latents": video data [B, C, T, H, W] or pre-encoded latents
        - "t5_text_embeddings": text condition [B, L, D]
        - "images" or "image_latents": conditioning image [B, C, 1, H, W] or pre-encoded

        Returns:
            raw_state, latent_state, condition (with y_B_C_T_H_W), uncondition
        """
        config = self.config

        # Process video data (same as T2V)
        if IS_PROCESSED_KEY not in data_batch or not data_batch[IS_PROCESSED_KEY]:
            if config.input_latent_key in data_batch:
                self._normalize_latent_inplace(data_batch)
                data_batch[config.input_data_key] = self.decode(data_batch[config.input_latent_key]).contiguous().float().clamp(-1, 1)
                data_batch[IS_PREPROCESSED_KEY] = True

            self._normalize_video_inplace(data_batch)
            data_batch[config.input_latent_key] = self.encode(data_batch[config.input_data_key]).contiguous().float()
            data_batch[IS_PROCESSED_KEY] = True

        raw_state = data_batch[config.input_data_key]
        latent_state = data_batch[config.input_latent_key]

        # Process image conditioning
        if config.input_image_latent_key in data_batch:
            image_latents = data_batch[config.input_image_latent_key].to(device="cuda")
        elif config.input_image_key in data_batch:
            # Encode image on the fly
            image = data_batch[config.input_image_key].to(device="cuda")
            if image.ndim == 4:
                # [B, C, H, W] -> [B, C, 1, H, W]
                image = image.unsqueeze(2)
            with torch.no_grad():
                image_latents = self.encode(image).contiguous().float()
        else:
            # Fallback: use first frame of video as image condition
            video = data_batch[config.input_data_key]
            with torch.no_grad():
                first_frame = video[:, :, :1, :, :]
                image_latents = self.encode(first_frame).contiguous().float()

        # Build y_B_C_T_H_W conditioning
        state_shape = latent_state.shape[1:]  # (C, T, H, W)
        y_B_C_T_H_W = self._build_image_condition(image_latents, state_shape)

        # Text condition
        if self.neg_embed is not None:
            data_batch["neg_t5_text_embeddings"] = repeat(
                self.neg_embed.to(**self.tensor_kwargs), "l d -> b l d", b=data_batch["t5_text_embeddings"].shape[0]
            )
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition = condition.edit_data_type(DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.VIDEO)

        # Attach y_B_C_T_H_W to condition for use in denoise()
        condition.y_B_C_T_H_W = y_B_C_T_H_W.to(**self.tensor_kwargs)
        uncondition.y_B_C_T_H_W = y_B_C_T_H_W.to(**self.tensor_kwargs)

        return raw_state, latent_state, condition, uncondition

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        time: torch.Tensor,
        condition: TextCondition,
        net_type: Literal["teacher", "fake_score", "student"] = "teacher",
    ) -> DenoisePrediction:
        """
        Denoise with image conditioning.
        Passes y_B_C_T_H_W to the network's forward method.
        """
        if time.ndim == 1:
            time_B_T = repeat(time, "b -> b 1")
        elif time.ndim == 2:
            time_B_T = time
        else:
            raise ValueError(f"time shape {time.shape} is not supported")
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")

        c_skip, c_out, c_in, c_noise = self.scaling(trigflow_t=time_B_1_T_1_1)

        net = {"student": self.net, "teacher": self.net_teacher, "fake_score": self.net_fake_score}[net_type]

        # Extract y_B_C_T_H_W from condition (attached in get_data_and_condition)
        cond_dict = condition.to_dict()
        y_B_C_T_H_W = getattr(condition, "y_B_C_T_H_W", None)

        net_output = net(
            x_B_C_T_H_W=(xt_B_C_T_H_W * c_in).to(**self.tensor_kwargs),
            timesteps_B_T=c_noise.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            y_B_C_T_H_W=y_B_C_T_H_W,
            **cond_dict,
        ).float()

        x0_pred = c_skip * xt_B_C_T_H_W + c_out * net_output
        F_pred = (torch.cos(time_B_1_T_1_1) * xt_B_C_T_H_W - x0_pred) / torch.sin(time_B_1_T_1_1)
        return DenoisePrediction(x0=x0_pred, F=F_pred)

    def student_F_withT(self, xt_B_C_T_H_W, time, condition: TextCondition):
        """JVP forward with image conditioning."""
        xt, t_xt = xt_B_C_T_H_W
        time_val, t_time = time

        if time_val.ndim == 1:
            time_B_T = rearrange(time_val, "b -> b 1")
            t_time_B_T = rearrange(t_time, "b -> b 1")
        elif time_val.ndim == 2:
            time_B_T = time_val
            t_time_B_T = t_time
        else:
            raise ValueError(f"time shape {time_val.shape} is not supported")

        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        t_time_B_1_T_1_1 = rearrange(t_time_B_T, "b t -> b 1 t 1 1")

        (c_skip, c_out, c_in, c_noise), (t_c_skip, t_c_out, t_c_in, t_c_noise) = torch.func.jvp(
            self.scaling, (time_B_1_T_1_1,), (t_time_B_1_T_1_1,)
        )

        def _process_input(xt, c_in):
            return xt * c_in

        x, t_x = torch.func.jvp(_process_input, (xt, c_in), (t_xt, t_c_in))

        # Extract y_B_C_T_H_W from condition
        y_B_C_T_H_W = getattr(condition, "y_B_C_T_H_W", None)

        cond_dict = condition.to_dict()
        net_output, t_net_output = self.net(
            x_B_C_T_H_W=(x.to(**self.tensor_kwargs), t_x.to(**self.tensor_kwargs)),
            timesteps_B_T=(
                c_noise.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
                t_c_noise.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            ),
            y_B_C_T_H_W=y_B_C_T_H_W,
            **cond_dict,
            withT=True,
        )
        net_output, t_net_output = net_output.float(), t_net_output.float()

        def _process_output(xt, net_out, c_skip, c_out, time):
            x0_pred = c_skip * xt + c_out * net_out
            F_pred = (torch.cos(time) * xt - x0_pred) / torch.sin(time)
            return F_pred

        F, t_F = torch.func.jvp(
            _process_output,
            (xt, net_output, c_skip, c_out, time_B_1_T_1_1),
            (t_xt, t_net_output, t_c_skip, t_c_out, t_time_B_1_T_1_1),
        )
        return (F, t_F.detach())

    def backward_simulation(self, condition, x_B_C_T_H_W_size, n_steps, with_grad=False):
        """Backward simulation with image conditioning passed through."""
        G_time_B_1 = math.pi / 2 * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda")
        x = torch.randn(x_B_C_T_H_W_size, device="cuda")
        x = self.sync(x)
        t_traj, x_traj = [G_time_B_1], [x]

        for i in range(n_steps - 1):
            if not self.config.dmd_fix_timesteps:
                G_time_B_1 = torch.minimum(self.draw_training_time_D((x_B_C_T_H_W_size[0], 1)), G_time_B_1)
                G_time_B_1 = self.sync(G_time_B_1)
                t_traj.append(G_time_B_1)
            else:
                backward_t = self.config.backward_timesteps[i]
                G_time_B_1 = backward_t * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda")
                t_traj.append(G_time_B_1)
        t_traj.append(0 * G_time_B_1)

        for step, (t_cur, t_next) in enumerate(zip(t_traj[:-1], t_traj[1:])):
            context_fn = torch.enable_grad if with_grad and step == n_steps - 1 else torch.no_grad
            with context_fn():
                x = self.denoise(x, t_cur, condition, net_type="student").x0.float()
            if step < n_steps - 1:
                x = torch.cos(rearrange(t_next, "b 1 -> b 1 1 1 1")) * x + torch.sin(
                    rearrange(t_next, "b 1 -> b 1 1 1 1")
                ) * self.sync(torch.randn_like(x))
            x_traj.append(x.detach())
        return x, (t_traj, x_traj)
