# Experiment configuration for Wan2.2-Fun-5B I2V SLA training.
#
# Usage:
#   torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
#       --config=rcm/configs/registry_sla_i2v.py -- \
#       experiment="wan2pt2_fun_5B_i2v_SLA"

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyDict


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
    )


WAN2PT2_FUN_5B_I2V_SLA: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "standard"},
            {"override /data_train": "webdataset_i2v"},
            {"override /model": "fsdp_i2v_sla"},
            {"override /net": "wan2pt2_fun_5B_i2v"},
            {"override /net_teacher": "wan2pt2_fun_5B_i2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_sla",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="SLA_Wan2.2_Fun",
            name="wan2pt2_fun_5B_i2v_SLA",
        ),
        optimizer=dict(
            lr=5e-6,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                sla_topk=0.1,
                fsdp_shard_size=8,
                resolution="720p",
                timestep_shift=5,
                state_t=21,
                grad_clip=False,
                loss_scale=1.0,
                precision="bfloat16",

                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                teacher_ckpt="assets/checkpoints/Wan2.2-Fun-5B.dcp",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                teacher_guidance=5.0,
                p_t=dict(p_mean=1.5, p_std=1.6),

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
            batch_size=2,
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

job_list = [WAN2PT2_FUN_5B_I2V_SLA]
for job in job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))
