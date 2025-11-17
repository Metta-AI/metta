"""Arena Basic Easy Shaped recipe targeting the SmolLLM policy."""

from __future__ import annotations

from typing import Any, Optional

import metta.cogworks.curriculum as cc
from metta.agent.policies.smollm import SmolLLMConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.tools.sweep import SweepTool
from mettagrid import MettaGridConfig
from recipes.prod.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    mettagrid as _base_mettagrid,
    play,
    replay,
    simulations,
    sweep as _arena_sweep,
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
    """ALL SmolLM configuration in ONE place.

    Returns: (policy_config, trainer_updates, env_updates, num_agents)
    """
    # Environment setup
    num_agents = 24

    # Policy architecture configuration
    policy_config = {
        "model_name": model_name or "HuggingFaceTB/SmolLM-360M",
        "attn_implementation": "flash_attention_2",
        "torch_dtype": "bfloat16",
        "mem_len": int(mem_len) if mem_len is not None else 16,
    }

    # Trainer configuration
    trainer_updates = {
        "compile": False,
        "batch_size": 131072,
        "minibatch_size": 4096,
        "bptt_horizon": 16,
    }

    # Environment configuration
    env_updates = {
        "forward_pass_minibatch_target_size": 4096,
        "auto_workers": False,
        "num_workers": 1,
        "async_factor": 1,
    }

    return policy_config, trainer_updates, env_updates, num_agents


def mettagrid(num_agents: int = 3) -> MettaGridConfig:
    """SmolLLM-optimized arena with fewer agents to reduce memory requirements."""
    return _base_mettagrid(num_agents=num_agents)


def train(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    model_name: Optional[str] = None,
    mem_len: Optional[int] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
):
    """Train SmolLLM with optimized defaults for memory-constrained environments."""
    # Get all SmolLM configuration from ONE place
    policy_config, trainer_updates, env_updates, num_agents = _smollm_config(
        model_name, mem_len
    )

    # Apply policy architecture
    if policy_architecture is None:
        policy_architecture = SmolLLMConfig(**policy_config)

    # Curriculum
    resolved_curriculum = curriculum or make_curriculum(
        arena_env=mettagrid(num_agents=num_agents),
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    # Create base tool
    tool = base_train(
        curriculum=resolved_curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    # Apply all configuration overrides
    tool.trainer = tool.trainer.model_copy(update=trainer_updates)
    tool.training_env = tool.training_env.model_copy(update=env_updates)

    return tool


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    *,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """SmolLLM-friendly arena curriculum that avoids legacy converter fields."""

    env = arena_env or mettagrid()
    tasks = cc.bucketed(env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return tasks.to_curriculum(algorithm_config=algorithm_config)


def sweep(
    sweep_name: str,
    **kwargs: object,
) -> SweepTool:
    """Expose the canonical arena sweep for SmolLLM recipes."""

    return _delegate_sweep(sweep_name, **kwargs)


def sweep_async_progressive(
    sweep_name: str,
    **kwargs: object,
) -> SweepTool:
    """Backward-compatible alias retained for historical CLI invocations."""

    return _delegate_sweep(sweep_name, **kwargs)


def _delegate_sweep(sweep_name: str, **kwargs: object) -> SweepTool:
    return _arena_sweep(sweep_name, **kwargs)
