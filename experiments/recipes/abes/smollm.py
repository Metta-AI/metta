"""Arena Basic Easy Shaped recipe targeting the SmolLLM policy."""

from __future__ import annotations

import importlib.util
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
from experiments.recipes.abes import _train_clamp
from metta.agent.policies.smollm import SmolLLMConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.tools.train import TrainTool

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
            "token_stride": 2,
            "actor_head_rank": 64,
            "value_head_rank": 8,
        }
        config_kwargs["model_name"] = model_name or "HuggingFaceTB/SmolLM2-135M"
        policy_architecture = SmolLLMConfig(**config_kwargs)

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    return _apply_smollm_defaults(tool)


SMOLLM_LIMITS = _train_clamp.ClampLimits(
    batch_cap=131_072,
    minibatch_cap=4_096,
    bptt_cap=4,
    forward_pass_cap=2_048,
    disable_compile=True,
    auto_workers=False,
    num_workers_cap=1,
    async_factor_cap=1,
)


def _apply_smollm_defaults(tool: TrainTool) -> TrainTool:
    """Clamp heavy training defaults to keep SmolLLM within memory limits."""

    return _train_clamp.clamp_training_resources(tool, SMOLLM_LIMITS)


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
