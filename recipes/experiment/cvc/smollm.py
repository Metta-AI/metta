"""CoGs vs Clips recipe targeting the SmolLLM policy."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from metta.agent.policies.smollm import SmolLLMConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import train as base_train

__all__ = ["train"]


def _smollm_config(
    model_name: Optional[str] = None,
    mem_len: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return SmolLLM policy + trainer env tweaks for constrained memory."""
    policy_config = {
        "model_name": model_name or "HuggingFaceTB/SmolLM-360M",
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
        "mem_len": int(mem_len) if mem_len is not None else 16,
    }

    trainer_updates = {
        "compile": False,
        "batch_size": 262144,
        "minibatch_size": 8192,
        "bptt_horizon": 16,
    }

    env_updates = {
        "forward_pass_minibatch_target_size": 16384,
        "auto_workers": False,
        "num_workers": 1,
        "async_factor": 1,
    }

    return policy_config, {"trainer": trainer_updates, "training_env": env_updates}


def train(
    *,
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    mission: Optional[str] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    model_name: Optional[str] = None,
    mem_len: Optional[int] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Train SmolLLM on CoGs vs Clips with tuned defaults."""
    policy_config, updates = _smollm_config(model_name, mem_len)

    if policy_architecture is None:
        policy_architecture = SmolLLMConfig(**policy_config)

    tool = base_train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        mission=mission,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )

    # Override policy/trainer/env for SmolLLM defaults.
    tool.policy_architecture = policy_architecture
    tool.trainer = tool.trainer.model_copy(update=updates["trainer"])
    tool.training_env = tool.training_env.model_copy(update=updates["training_env"])

    return tool
