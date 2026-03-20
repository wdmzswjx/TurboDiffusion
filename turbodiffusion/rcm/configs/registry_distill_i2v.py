# Registry for Wan2.2-Fun-5B I2V rCM distillation training.
#
# Usage:
#   torchrun --nproc_per_node=4 -m scripts.train \
#       --config=rcm/configs/registry_distill_i2v.py -- \
#       experiment="wan2pt2_fun_5B_i2v_rCM"

from typing import Any, List

import attrs
import torch
from hydra.core.config_store import ConfigStore

from imaginaire import config
from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils.config_helper import import_all_modules_from_package

# Reuse existing defaults
from rcm.configs.defaults.trainer import register_trainer
from rcm.configs.defaults.checkpoint import register_checkpoint
from rcm.configs.defaults.ema import register_ema
from rcm.configs.defaults.optimizer import register_optimizer, register_optimizer_fake_score
from rcm.configs.defaults.scheduler import register_scheduler
from rcm.configs.defaults.conditioner import register_conditioner
from rcm.configs.defaults.callbacks import register_callbacks
from rcm.configs.defaults.ckpt_type import register_ckpt_type
from rcm.configs.defaults.tokenizer import register_tokenizer

# I2V-specific registrations
from rcm.configs.defaults.net_wan2pt2_fun import (
    register_net_wan2pt2_fun,
    register_net_teacher_wan2pt2_fun,
    register_net_fake_score_wan2pt2_fun,
)
from rcm.models.i2v_model_distill_rcm import I2VDistillConfig_rCM, I2VDistillModel_rCM
from rcm.datasets.webdataset_i2v import create_dataloader_i2v


# I2V model config
FSDP_CONFIG_I2V_DISTILL_RCM = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(I2VDistillModel_rCM)(config=I2VDistillConfig_rCM(fsdp_shard_size=8), _recursive_=False),
)

# I2V dataloader
WEBDATASET_I2V_LOADER = L(create_dataloader_i2v)(
    tar_path_pattern="/path/to/dataset/shard_*.tar",
    batch_size=1,
    num_workers=8,
    shuffle_buffer=1000,
    prefetch_factor=2,
)


def register_model_i2v():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="fsdp_i2v_distill_rcm", node=FSDP_CONFIG_I2V_DISTILL_RCM)


def register_dataloader_i2v():
    cs = ConfigStore.instance()
    DUMMY = L(torch.utils.data.DataLoader)(dataset=lambda: torch.utils.data.TensorDataset(torch.empty(0, 1), torch.empty(0)))
    cs.store(group="data_train", package="dataloader_train", name="dummy", node=DUMMY)
    cs.store(group="data_train", package="dataloader_train", name="webdataset_i2v", node=WEBDATASET_I2V_LOADER)
    cs.store(group="data_val", package="dataloader_val", name="dummy", node=DUMMY)


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"trainer": "standard"},
            {"data_train": "dummy"},
            {"data_val": "dummy"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"callbacks": "basic"},
            {"checkpoint": "local"},
            {"ckpt_type": "dcp"},
            {"model": "fsdp_i2v_distill_rcm"},
            {"net": None},
            {"net_teacher": None},
            {"net_fake_score": None},
            {"optimizer_fake_score": "fusedadamw"},
            {"conditioner": "text_nodrop"},
            {"ema": "power"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "rcm_i2v"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 100
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # Register all config groups
    register_trainer()
    register_dataloader_i2v()
    register_optimizer()
    register_optimizer_fake_score()
    register_scheduler()
    register_callbacks()
    register_checkpoint()
    register_ckpt_type()
    register_model_i2v()
    register_net_wan2pt2_fun()
    register_net_teacher_wan2pt2_fun()
    register_net_fake_score_wan2pt2_fun()
    register_conditioner()
    register_ema()
    register_tokenizer()

    # Import experiment configs
    import_all_modules_from_package("rcm.configs.experiments.rcm_i2v", reload=True)
    return c
