from experiments.recipes import arena_basic_easy_shaped as base
from metta.agent.policies.trxl_nvidia import TRXLNvidiaConfig
from metta.agent.policies.transformer import TransformerPolicyConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.trainer_config import OptimizerConfig

make_mettagrid = base.make_mettagrid
make_curriculum = base.make_curriculum
make_evals = base.make_evals
play = base.play
replay = base.replay
evaluate = base.evaluate
evaluate_in_sweep = base.evaluate_in_sweep
sweep_async_progressive = base.sweep_async_progressive

DEFAULT_LR = OptimizerConfig.model_fields["learning_rate"].default


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    if policy_architecture is None:
        policy_architecture = TransformerPolicyConfig(transformer=TRXLNvidiaConfig())

    tool = base.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    if isinstance(policy_architecture, TransformerPolicyConfig):
        hint = policy_architecture.learning_rate_hint
        optimizer = tool.trainer.optimizer
        if hint is not None and optimizer.learning_rate == DEFAULT_LR:
            optimizer.learning_rate = hint

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
