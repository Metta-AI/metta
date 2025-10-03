"""Arena Basic Easy Shaped recipe targeting the SmolLLM policy."""

from __future__ import annotations

from typing import Optional

from experiments.recipes import arena_basic_easy_shaped as base
from metta.agent.policies.smollm import SmolLLMConfig
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
    freeze_llm: bool = True,
    model_name: Optional[str] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
):
    if policy_architecture is None:
        config_kwargs = {"freeze_llm": freeze_llm}
        if model_name is not None:
            config_kwargs["model_name"] = model_name
        policy_architecture = SmolLLMConfig(**config_kwargs)

    return base.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )


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
