"""Arena Basic Easy Shaped recipe targeting the SmolLLM policy."""

from __future__ import annotations

from typing import Any, Optional

from metta.agent.policies.smollm import SmolLLMConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from recipes.prod.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
)
from recipes.prod.arena_basic_easy_shaped import (
    sweep as _arena_sweep,
)
from recipes.prod.arena_basic_easy_shaped import (
    train as base_train,
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
    "sweep",
    "train",
]


def _smollm_config(
    model_name: Optional[str] = None,
    mem_len: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    """SmolLM configuration returning policy, trainer, env updates, and agent count."""
    num_agents = 24

    policy_config = {
        "model_name": model_name or "HuggingFaceTB/SmolLM-360M",
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
        "mem_len": int(mem_len) if mem_len is not None else 16,
    }

    trainer_updates = {
        "compile": False,
        "batch_size": 131072,
        "minibatch_size": 4096,
        "bptt_horizon": 16,
    }

    env_updates = {
        "forward_pass_minibatch_target_size": 8192,
        "auto_workers": False,
        "num_workers": 1,
        "async_factor": 1,
    }

    return policy_config, trainer_updates, env_updates, num_agents


def train(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    model_name: Optional[str] = None,
    mem_len: Optional[int] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Train SmolLLM with optimized defaults for memory-constrained environments."""
    policy_config, trainer_updates, env_updates, _num_agents = _smollm_config(model_name, mem_len)

    if policy_architecture is None:
        policy_architecture = SmolLLMConfig(**policy_config)

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    tool.trainer = tool.trainer.model_copy(update=trainer_updates)
    tool.training_env = tool.training_env.model_copy(update=env_updates)

    return tool


def sweep(
    sweep_name: str,
    **kwargs: object,
) -> SweepTool:
    """Expose the canonical arena sweep for SmolLLM recipes."""

    return _arena_sweep(sweep_name, **kwargs)


def sweep_async_progressive(
    sweep_name: str,
    **kwargs: object,
) -> SweepTool:
    """Backward-compatible alias retained for historical CLI invocations."""

    return sweep(sweep_name, **kwargs)
