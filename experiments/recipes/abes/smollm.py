"""Arena Basic Easy Shaped recipe targeting the SmolLLM policy."""

from __future__ import annotations

import importlib.util
from typing import Optional

from experiments.recipes import arena_basic_easy_shaped as base
from metta.agent.policies.smollm import SmolLLMConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig

make_mettagrid = base.make_mettagrid
make_curriculum = base.make_curriculum
make_evals = base.make_evals
play = base.play
replay = base.replay
evaluate = base.evaluate
evaluate_in_sweep = base.evaluate_in_sweep
sweep_async_progressive = base.sweep_async_progressive


_FLASH_ATTENTION_AVAILABLE = importlib.util.find_spec("flash_attn") is not None


def _select_attn_implementation(attn_implementation: Optional[str]) -> Optional[str]:
    """Prefer FlashAttention when installed; otherwise fall back to eager attention."""

    if attn_implementation is not None:
        return attn_implementation
    if _FLASH_ATTENTION_AVAILABLE:
        return "flash_attention_2"
    return None


def train(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    freeze_llm: bool = True,
    model_name: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
):
    if policy_architecture is None:
        config_kwargs = {
            "freeze_llm": freeze_llm,
            "attn_implementation": _select_attn_implementation(attn_implementation),
        }
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
