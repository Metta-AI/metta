from metta.common.wandb.wandb_context import WandbConfig


def wandb_config(
    run: str | None = None,
) -> WandbConfig:
    cfg = WandbConfig(
        enabled=True,
        project="metta",
        entity="metta-research",
    )
    if run:
        cfg.name = run
        cfg.group = run
        cfg.run_id = run
        cfg.data_dir = f"./train_dir/{run}"
    return cfg
