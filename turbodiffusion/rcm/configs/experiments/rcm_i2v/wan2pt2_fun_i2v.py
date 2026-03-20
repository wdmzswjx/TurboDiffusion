# Experiment configuration for Wan2.2-Fun-5B I2V rCM distillation training.
#
# Usage:
#   torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
#       --config=rcm/configs/registry_distill_i2v.py -- \
#       experiment="wan2pt2_fun_5B_i2v_rCM"

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from rcm.utils.timestep_utils import LogNormal, UniformShift


def build_debug_run(job):
    return dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=dict(
            max_iter=25,
            logging_iter=2,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=6, num_samples=2),
                every_n_sample_ema=dict(every_n=6, num_samples=2),
            ),
        ),
        checkpoint=dict(
            save_iter=10,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(tangent_warmup=8),
        ),
    )


WAN2PT2_FUN_5B_I2V_RCM: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset_i2v"},
            {"override /model": "fsdp_i2v_distill_rcm"},
            {"override /net": "wan2pt2_fun_5B_i2v_jvp"},
            {"override /net_teacher": "wan2pt2_fun_5B_i2v"},
            {"override /net_fake_score": "wan2pt2_fun_5B_i2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {"override /optimizer_fake_score": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="rCM_Wan2.2_Fun",
            name="wan2pt2_fun_5B_i2v_rCM",
        ),
        optimizer=dict(
            lr=1e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                # Loss
                loss_scale=100.0,
                loss_scale_dmd=1.0,

                # FSDP
                fsdp_shard_size=8,

                # Resolution
                resolution="720p",

                # Timestep sampling
                p_G=L(LogNormal)(p_mean=1.0, p_std=1.6),
                p_D=L(UniformShift)(shift=5.0),

                # Training dynamics
                max_simulation_steps_fake=4,
                state_t=21,
                sigma_max=200,
                grad_clip=False,
                rectified_flow_t_scaling_factor=1000.0,
                student_update_freq=5,
                tangent_warmup=500,
                precision="bfloat16",

                # Fake score optimizer
                optimizer_fake_score=dict(
                    lr=2e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),

                # Paths
                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                teacher_ckpt="assets/checkpoints/Wan2.2-Fun-5B.dcp",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                teacher_guidance=5.0,

                # Network SAC
                net=dict(
                    sac_config=dict(mode="block_wise"),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=500, num_samples=3, run_at_start=True),
                every_n_sample_ema=dict(every_n=500, num_samples=3),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/Wan2.2_Fun_5B_720p_I2V/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

job_list = [WAN2PT2_FUN_5B_I2V_RCM]
for job in job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))
