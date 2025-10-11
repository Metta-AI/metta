from typing import Any, Optional

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
from metta.agent.policies.cortex import CortexBaseConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.policy import PolicyArchitecture
from metta.tools.train import TrainTool
from mettagrid.util.module import load_symbol
from metta.cogworks.curriculum.curriculum import CurriculumConfig


def _override_cortex_stack(
    policy_cfg: CortexBaseConfig, stack: Any
) -> CortexBaseConfig:
    """Replace the stack inside any CortexTD components in the policy config."""
    components = []
    for comp in policy_cfg.components:
        if isinstance(comp, CortexTDConfig):
            new_comp = comp.model_copy()
            new_comp.stack = stack
            components.append(new_comp)
        else:
            components.append(comp)
    policy_cfg.components = components
    return policy_cfg


def train(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: Optional[PolicyArchitecture] = None,
    stack: Any | None = None,
    stack_builder: Optional[str] = None,
    stack_cfg: Optional[dict] = None,
) -> TrainTool:
    # Default to Cortex policy and apply optional stack overrides
    if policy_architecture is None:
        policy_architecture = CortexBaseConfig()
        if stack is not None:
            policy_architecture = _override_cortex_stack(policy_architecture, stack)
        elif stack_builder is not None:
            builder = load_symbol(stack_builder)
            built_stack = builder(**(stack_cfg or {}))
            policy_architecture = _override_cortex_stack(
                policy_architecture, built_stack
            )

    return base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
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
