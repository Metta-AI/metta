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
from metta.agent.policies.vit_sliding_trans import ViTSlidingTransConfig
from metta.agent.policy import PolicyArchitecture


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    policy = policy_architecture or ViTSlidingTransConfig()
    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy,
    )

    # Adjust optimizer + trainer settings for the smaller sliding transformer architecture
    optimizer = tool.trainer.optimizer
    optimizer.learning_rate = 0.001153637
    tool.trainer.batch_size = 131072
    tool.trainer.minibatch_size = 4096
    tool.training_env.forward_pass_minibatch_target_size = 1024

    # Disable periodic evaluation by default for this recipe
    tool.evaluator.epoch_interval = 0

    return tool


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
]
