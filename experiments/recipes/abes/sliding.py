from experiments.recipes import arena_basic_easy_shaped as base
from metta.agent.policies.vit_sliding_trans import ViTSlidingTransConfig
from metta.agent.policy import PolicyArchitecture

make_mettagrid = base.make_mettagrid
make_curriculum = base.make_curriculum
make_evals = base.make_evals
play = base.play
replay = base.replay
evaluate = base.evaluate
evaluate_in_sweep = base.evaluate_in_sweep
sweep_async_progressive = base.sweep_async_progressive


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    policy = policy_architecture or ViTSlidingTransConfig()
    tool = base.train(
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
    "make_mettagrid",
    "make_curriculum",
    "make_evals",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep_async_progressive",
    "train",
]
