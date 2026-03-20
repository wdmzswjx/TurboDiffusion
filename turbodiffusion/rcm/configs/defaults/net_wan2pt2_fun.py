# Network configuration for Wan2.2-Fun-5B (I2V model)

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict

from rcm.networks.wan2pt2 import WanModel
from rcm.networks.wan2pt1_jvp import WanModel_JVP

# Wan2.2-Fun-5B architecture: dim=3072, 36 layers, 24 heads
# in_dim=36 for I2V (16 latent channels + 4 mask channels + 16 image latent channels)
wan2pt2_fun_5B_net_args = dict(
    dim=3072,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=36,
    num_heads=24,
    num_layers=36,
    out_dim=16,
    text_len=512,
)

# Standard forward-only network (for teacher & fake_score)
WAN2PT2_FUN_5B_I2V: LazyDict = L(WanModel)(**wan2pt2_fun_5B_net_args, model_type="i2v")

# JVP-enabled network (for student in rCM distillation)
WAN2PT2_FUN_5B_I2V_JVP: LazyDict = L(WanModel_JVP)(**wan2pt2_fun_5B_net_args, model_type="i2v")


def register_net_wan2pt2_fun():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt2_fun_5B_i2v", node=WAN2PT2_FUN_5B_I2V)
    cs.store(group="net", package="model.config.net", name="wan2pt2_fun_5B_i2v_jvp", node=WAN2PT2_FUN_5B_I2V_JVP)


def register_net_teacher_wan2pt2_fun():
    cs = ConfigStore.instance()
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt2_fun_5B_i2v", node=WAN2PT2_FUN_5B_I2V)


def register_net_fake_score_wan2pt2_fun():
    cs = ConfigStore.instance()
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt2_fun_5B_i2v", node=WAN2PT2_FUN_5B_I2V)
