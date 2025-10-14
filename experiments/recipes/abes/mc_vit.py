from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep_async_progressive,
)
from experiments.recipes.arena_basic_easy_shaped import (
    train as base_train,
)
from metta.agent.policies.mc_vit import MCViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.loss.loss_config import LossConfig
from metta.rl.loss.mc_ppo import MCPPOConfig
from metta.rl.trainer_config import TrainerConfig
from metta.common.wandb.context import WandbConfig
from metta.rl.system_config import SystemConfig


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    return base_train(
        trainer_cfg=TrainerConfig(
            losses=LossConfig(loss_configs={"mc_ppo": MCPPOConfig()})
        ),
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture or MCViTDefaultConfig(),
        wandb=WandbConfig.Off(),
        stats_server_uri=None,
        system=SystemConfig(local_only=True),
    )


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep_async_progressive",
    "train",
]
