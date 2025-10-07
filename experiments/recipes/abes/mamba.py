from typing import Optional

from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep_async_progressive,
    train as base_train,
)
from metta.agent.components.mamba import MambaBackboneConfig
from metta.agent.policies.mamba_sliding import MambaSlidingConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.tools.train import TrainTool

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
    "train_mamba2",
]


def _set_ssm_layer(
    policy_architecture: PolicyArchitecture, layer: str
) -> PolicyArchitecture:
    for component in policy_architecture.components:
        if isinstance(component, MambaBackboneConfig):
            next_cfg = dict(component.ssm_cfg) if component.ssm_cfg else {}
            next_cfg["layer"] = layer
            component.ssm_cfg = next_cfg
    return policy_architecture


def train(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    ssm_layer: str | None = "Mamba1",
) -> TrainTool:
    policy = policy_architecture or MambaSlidingConfig()
    if ssm_layer is not None:
        policy = _set_ssm_layer(policy, ssm_layer)

    return base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy,
    )


def train_mamba2(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
) -> TrainTool:
    return train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
        ssm_layer="Mamba2",
    )
