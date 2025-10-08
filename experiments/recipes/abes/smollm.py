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
        }
        config_kwargs["model_name"] = model_name or "HuggingFaceTB/SmolLM2-135M"
        policy_architecture = SmolLLMConfig(**config_kwargs)

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    return _apply_smollm_defaults(tool)


def _apply_smollm_defaults(tool: TrainTool) -> TrainTool:
    """Clamp heavy training defaults to keep SmolLLM within memory limits."""

    trainer_updates = {}
    if tool.trainer.batch_size > 4096:
        trainer_updates["batch_size"] = 4096
    if tool.trainer.minibatch_size > 1024:
        trainer_updates["minibatch_size"] = 1024
    if tool.trainer.bptt_horizon > 4:
        trainer_updates["bptt_horizon"] = (
            4  # keep segments >= agents without inflating batch size
        )
    if tool.trainer.compile:
        trainer_updates["compile"] = False
    if trainer_updates:
        tool.trainer = tool.trainer.model_copy(update=trainer_updates)

    env_updates = {}
    if tool.training_env.forward_pass_minibatch_target_size > 1024:
        env_updates["forward_pass_minibatch_target_size"] = 1024
    if tool.training_env.async_factor > 1:
        env_updates["async_factor"] = 1
    if tool.training_env.auto_workers:
        env_updates["auto_workers"] = False
    if tool.training_env.num_workers > 1:
        env_updates["num_workers"] = 1
    if env_updates:
        tool.training_env = tool.training_env.model_copy(update=env_updates)

    return tool


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
