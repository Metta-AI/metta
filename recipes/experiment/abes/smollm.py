"""Arena Basic Easy Shaped recipe targeting the SmolLLM policy."""

from __future__ import annotations

from typing import Any, Optional

from metta.agent.policies.smollm import SmolLLMConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.training.scheduler import HyperUpdateRule, LossRunGate, SchedulerConfig
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

    return policy_config, trainer_updates, env_updates, num_agents


def _kickstarter_config() -> tuple[dict[str, Any], SchedulerConfig]:
    """Kickstarter configuration for SmolLLM on Arena Basic Easy Shaped."""
    loss_config = LossesConfig()
    loss_config.ppo_critic.enabled = False
    loss_config.ppo_actor.enabled = False
    loss_config.ppo.enabled = True
    loss_config.kickstarter.enabled = True
    loss_config.kickstarter.teacher_uri = (
        "s3://softmax-public/policies/av.sliced.mb.11.22.110.ctrl/av.sliced.mb.11.22.110.ctrl:v9900.mpt"
    )

    trainer_updates: dict[str, Any] = {
        "losses": loss_config,
    }

    scheduler = SchedulerConfig(
        run_gates=[
            LossRunGate(loss_instance_name="ppo", phase="rollout", begin_at_step=1_000_000_000),
            LossRunGate(
                loss_instance_name="kickstarter",
                phase="rollout",
                end_at_step=1_000_000_000,
            ),
            LossRunGate(
                loss_instance_name="kickstarter",
                phase="train",
                end_at_step=1_000_000_000,
            ),
        ],
        rules=[
            HyperUpdateRule(
                loss_instance_name="kickstarter",
                attr_path="action_loss_coef",
                mode="progress",
                style="linear",
                start_value=0.6,
                end_value=0.0,
                start_agent_step=500_000_000,
                end_agent_step=1_000_000_000,
            ),
            HyperUpdateRule(
                loss_instance_name="kickstarter",
                attr_path="value_loss_coef",
                mode="progress",
                style="linear",
                start_value=1.0,
                end_value=0.0,
                start_agent_step=500_000_000,
                end_agent_step=1_000_000_000,
            ),
            HyperUpdateRule(
                loss_instance_name="kickstarter",
                attr_path="teacher_lead_prob",
                mode="progress",
                style="linear",
                start_value=1.0,
                end_value=0.0,
                start_agent_step=30_000_000,
                end_agent_step=500_000_000,
            ),
        ],
    )

    return trainer_updates, scheduler


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
    kickstarter_trainer_updates, scheduler = _kickstarter_config()

    if policy_architecture is None:
        policy_architecture = SmolLLMConfig(**policy_config)

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    combined_trainer_updates: dict[str, Any] = {**trainer_updates, **kickstarter_trainer_updates}

    tool.trainer = tool.trainer.model_copy(update=combined_trainer_updates)
    tool.training_env = tool.training_env.model_copy(update=env_updates)
    tool.scheduler = scheduler

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
