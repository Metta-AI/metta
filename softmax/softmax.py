import uuid

from metta.common.wandb.wandb_context import WandbConfig


def wandb_config(
    run: str | None = None,
) -> WandbConfig:
    run_name = run or "run-" + str(uuid.uuid4())

    cfg = WandbConfig(
        enabled=True,
        project="metta",
        entity="metta-research",
        name=run_name,
        group=run_name,
        run_id=run_name,
        data_dir=f"./train_dir/{run_name}",
    )
    return cfg
