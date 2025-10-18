import metta.cogworks.curriculum as cc
from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep,
    train as base_train,
)
from metta.agent.policies.vit_consistent_dropout import ViTConsistentDropoutConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from mettagrid.config import ConverterConfig


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    return base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture or ViTConsistentDropoutConfig(),
    )


def train_shaped(
    *,
    rewards: bool = True,
    converters: bool = True,
    policy_architecture: PolicyArchitecture | None = None,
) -> TrainTool:
    """Training with shaped rewards and consistent dropout policy."""
    env_cfg = mettagrid()
    env_cfg.game.agent.rewards.inventory["heart"] = 1
    env_cfg.game.agent.rewards.inventory_max["heart"] = 100

    if rewards:
        env_cfg.game.agent.rewards.inventory.update(
            {
                "ore_red": 0.1,
                "battery_red": 0.8,
                "laser": 0.5,
                "armor": 0.5,
                "blueprint": 0.5,
            }
        )
        env_cfg.game.agent.rewards.inventory_max.update(
            {
                "ore_red": 1,
                "battery_red": 1,
                "laser": 1,
                "armor": 1,
                "blueprint": 1,
            }
        )

    if converters:
        altar = env_cfg.game.objects.get("altar")
        if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
            altar.input_resources["battery_red"] = 1

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env_cfg)),
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=policy_architecture or ViTConsistentDropoutConfig(),
    )


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep",
    "train",
    "train_shaped",
]
