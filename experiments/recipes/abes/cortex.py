import typing

import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policies.cortex
import cortex.config
import metta.agent.components.cortex
import metta.agent.policy
import metta.tools.train
import mettagrid.util.module
import metta.cogworks.curriculum.curriculum


def _override_cortex_stack(
    policy_cfg: metta.agent.policies.cortex.CortexBaseConfig, stack: typing.Any
) -> metta.agent.policies.cortex.CortexBaseConfig:
    """Replace the stack inside any CortexTD components in the policy config."""
    components = []
    for comp in policy_cfg.components:
        if isinstance(comp, metta.agent.components.cortex.CortexTDConfig):
            new_comp = comp.model_copy()
            # Accept a config or dict; require stack_cfg only
            if isinstance(stack, cortex.config.CortexStackConfig):
                new_comp.stack_cfg = stack
            elif isinstance(stack, dict):
                new_comp.stack_cfg = cortex.config.CortexStackConfig(**stack)
            else:
                raise TypeError(
                    "_override_cortex_stack expects a CortexStackConfig or dict"
                )
            components.append(new_comp)
        else:
            components.append(comp)
    policy_cfg.components = components
    return policy_cfg


def train(
    *,
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: typing.Optional[metta.agent.policy.PolicyArchitecture] = None,
    stack: typing.Any | None = None,
    stack_builder: typing.Optional[str] = None,
    stack_cfg: typing.Optional[dict] = None,
) -> metta.tools.train.TrainTool:
    # Default to Cortex policy and apply optional stack overrides
    if policy_architecture is None:
        policy_architecture = metta.agent.policies.cortex.CortexBaseConfig()
        if stack is not None:
            policy_architecture = _override_cortex_stack(policy_architecture, stack)
        elif stack_builder is not None:
            builder = mettagrid.util.module.load_symbol(stack_builder)
            built = builder(**(stack_cfg or {}))
            policy_architecture = _override_cortex_stack(policy_architecture, built)

    return experiments.recipes.arena_basic_easy_shaped.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )


if "mettagrid" not in globals():
    mettagrid = experiments.recipes.arena_basic_easy_shaped.mettagrid
if "make_curriculum" not in globals():
    make_curriculum = experiments.recipes.arena_basic_easy_shaped.make_curriculum
if "simulations" not in globals():
    simulations = experiments.recipes.arena_basic_easy_shaped.simulations
if "play" not in globals():
    play = experiments.recipes.arena_basic_easy_shaped.play
if "replay" not in globals():
    replay = experiments.recipes.arena_basic_easy_shaped.replay
if "evaluate" not in globals():
    evaluate = experiments.recipes.arena_basic_easy_shaped.evaluate
if "evaluate_in_sweep" not in globals():
    evaluate_in_sweep = experiments.recipes.arena_basic_easy_shaped.evaluate_in_sweep
if "sweep" not in globals():
    sweep = experiments.recipes.arena_basic_easy_shaped.sweep


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "train",
]
