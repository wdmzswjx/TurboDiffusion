# SPDX-License-Identifier: Apache-2.0
#
# I2V (Image-to-Video) SLA training model for Wan2.2-Fun-5B.
# Extends the T2V SLA model with image conditioning support.

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

import attrs
import torch
from einops import rearrange, repeat
from torch import Tensor

from imaginaire.lazy_config import LazyCall as L, LazyDict
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.utils import log, misc

from rcm.conditioner import DataType, TextCondition
from rcm.models.t2v_model_sla import (
    T2VConfig_SLA,
    T2VModel_SLA,
    DenoisePrediction,
    IS_PREPROCESSED_KEY,
    IS_PROCESSED_KEY,
    replace_attention_with_sla,
)
from rcm.networks.wan2pt2 import WanSelfAttention
from SLA import SparseLinearAttention


def replace_attention_with_sla_wan2pt2(model: torch.nn.Module, sla_topk: float):
    """Replace attention in Wan2.2 model with SLA."""
    for module in model.modules():
        if type(module) is WanSelfAttention:
            module.attn_op.local_attn = SparseLinearAttention(head_dim=module.head_dim, topk=sla_topk)


@attrs.define(slots=False)
class I2VConfig_SLA(T2VConfig_SLA):
    """Extends T2V SLA config with I2V-specific parameters."""

    input_image_key: str = "images"
    input_image_latent_key: str = "image_latents"
    state_ch: int = 16
    in_ch_with_image: int = 36
    mask_first_frame: bool = True


class I2VModel_SLA(T2VModel_SLA):
    """
    Image-to-Video SLA training model.

    Extends T2V SLA with image conditioning for the Wan2.2 I2V architecture.
    """

    def __init__(self, config: I2VConfig_SLA):
        super().__init__(config)

    def build_net(self, net_dict: LazyDict, replace_sla=False):
        """Build network, using Wan2.2-compatible SLA replacement."""
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            with torch.device(init_device):
                net = lazy_instantiate(net_dict)

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                net.init_weights()
            if replace_sla:
                log.info("Replacing attention with SLA (Wan2.2)")
                # Try Wan2.2 attention first, fall back to Wan2.1
                replace_attention_with_sla_wan2pt2(net, self.config.sla_topk)
                replace_attention_with_sla(net, self.config.sla_topk)

            if self.fsdp_device_mesh:
                from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
                from rcm.utils.dtensor_helper import broadcast_dtensor_model_states
                from torch.distributed._tensor.api import DTensor
                mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
                net.fully_shard(mesh=self.fsdp_device_mesh, mp_policy=mp_policy)
                net = fully_shard(net, mesh=self.fsdp_device_mesh, mp_policy=mp_policy, reshard_after_forward=True)
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    def _build_image_condition(
        self,
        image_latents: torch.Tensor,
        state_shape: tuple,
    ) -> torch.Tensor:
        """Build I2V conditioning tensor y_B_C_T_H_W."""
        B = image_latents.shape[0]
        C_lat, T, H, W = state_shape

        msk = torch.zeros(B, 4, T, H, W, device=image_latents.device, dtype=image_latents.dtype)
        if self.config.mask_first_frame:
            msk[:, :, 0, :, :] = 1.0

        img_lat = torch.zeros(B, C_lat, T, H, W, device=image_latents.device, dtype=image_latents.dtype)
        if image_latents.shape[2] == 1:
            img_lat[:, :, 0, :, :] = image_latents[:, :, 0, :, :]
        else:
            img_lat = image_latents

        y = torch.cat([msk, img_lat], dim=1)
        return y

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor, TextCondition, TextCondition]:
        """Process data batch for I2V SLA training."""
        config = self.config

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
            image = data_batch[config.input_image_key].to(device="cuda")
            if image.ndim == 4:
                image = image.unsqueeze(2)
            with torch.no_grad():
                image_latents = self.encode(image).contiguous().float()
        else:
            # Fallback: use first frame
            video = data_batch[config.input_data_key]
            with torch.no_grad():
                first_frame = video[:, :, :1, :, :]
                image_latents = self.encode(first_frame).contiguous().float()

        state_shape = latent_state.shape[1:]
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

        condition.y_B_C_T_H_W = y_B_C_T_H_W.to(**self.tensor_kwargs)
        uncondition.y_B_C_T_H_W = y_B_C_T_H_W.to(**self.tensor_kwargs)

        return raw_state, latent_state, condition, uncondition

    def training_step(self, data_batch: dict[str, torch.Tensor], iteration: int):
        """Training step with I2V conditioning."""
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)

        time_B_T = self.draw_training_time((x0_B_C_T_H_W.shape[0], 1))
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        x0_B_C_T_H_W, time_B_T, epsilon_B_C_T_H_W, condition = self.sync(x0_B_C_T_H_W, time_B_T, epsilon_B_C_T_H_W, condition)

        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        xt_B_C_T_H_W = (1 - time_B_1_T_1_1) * x0_B_C_T_H_W + time_B_1_T_1_1 * epsilon_B_C_T_H_W

        # Extract y_B_C_T_H_W from condition
        y_B_C_T_H_W = getattr(condition, "y_B_C_T_H_W", None)

        net_output = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(time_B_1_T_1_1 * self.config.rectified_flow_t_scaling_factor).squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            y_B_C_T_H_W=y_B_C_T_H_W,
            **condition.to_dict(),
        ).float()

        with torch.no_grad():
            teacher_output = self.net_teacher(
                x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
                timesteps_B_T=(time_B_1_T_1_1 * self.config.rectified_flow_t_scaling_factor).squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
                y_B_C_T_H_W=y_B_C_T_H_W,
                **condition.to_dict(),
            ).float()

        kendall_loss = self.config.loss_scale * ((net_output - teacher_output) ** 2).mean(dim=(1, 2, 3, 4))
        output_batch = {
            "x0": x0_B_C_T_H_W.detach().cpu(),
            "xt": xt_B_C_T_H_W.detach().cpu(),
            "F_pred": net_output.detach().cpu(),
            "teacher_F_pred": teacher_output.detach().cpu(),
        }

        kendall_loss = kendall_loss.mean()
        return output_batch, kendall_loss
