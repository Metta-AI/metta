from metta.agent.policies.gtrxl import GTrXLConfig
from metta.agent.policies.transformer import TransformerPolicyConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.policy_assets import OptimizerConfig
from recipes.prod.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep,
)
from recipes.prod.arena_basic_easy_shaped import (
    train as base_train,
)

DEFAULT_LR = OptimizerConfig.model_fields["learning_rate"].default


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
):
    if policy_architecture is None:
        policy_architecture = TransformerPolicyConfig(transformer=GTrXLConfig())

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    if isinstance(policy_architecture, TransformerPolicyConfig):
        hint = policy_architecture.learning_rate_hint
        asset = (tool.policy_assets or {}).get("primary")
        optimizer = getattr(asset, "optimizer", None) if asset is not None else None
        if hint is not None and optimizer is not None and optimizer.learning_rate == DEFAULT_LR:
            optimizer.learning_rate = hint

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
