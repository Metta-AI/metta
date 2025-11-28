"""Arena Basic Easy Shaped recipe targeting the SmolLLM policy."""

from __future__ import annotations

from typing import Any, Optional

from metta.agent.policies.smollm import SmolLLMConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import OptimizerConfig
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
    learning_rate: Optional[float] = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, float, float]:
    """SmolLM configuration returning policy, trainer, env updates, agent count, and KS loss coefficients."""
    num_agents = 24

    lr = float(learning_rate) if learning_rate is not None else 1e-3

    policy_config = {
        "model_name": model_name or "HuggingFaceTB/SmolLM-135M",
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
        "mem_len": int(mem_len) if mem_len is not None else 16,
    }

    trainer_updates = {
        "compile": False,
        "batch_size": 262144,
        "minibatch_size": 8192,
        "bptt_horizon": 16,
        "optimizer": OptimizerConfig(learning_rate=lr),
    }

    env_updates = {
        "forward_pass_minibatch_target_size": 16384,
        "auto_workers": False,
        "num_workers": 1,
        "async_factor": 1,
    }

    ks_act_loss_coef = 2.0
    ks_value_loss_coef = 1.0

    return policy_config, trainer_updates, env_updates, num_agents, ks_act_loss_coef, ks_value_loss_coef


def _kickstarter_config(ks_act_loss_coef: float, ks_value_loss_coef: float) -> tuple[dict[str, Any], SchedulerConfig]:
    """Kickstarter configuration for SmolLLM on Arena Basic Easy Shaped."""
    loss_config = LossesConfig()
    loss_config.ppo_critic.enabled = False
    loss_config.ppo_actor.enabled = False
    loss_config.ppo.enabled = True
    loss_config.kickstarter.enabled = True
    #loss_config.kickstarter.student_forward = True
    loss_config.kickstarter.action_loss_coef = ks_act_loss_coef
    loss_config.kickstarter.value_loss_coef = ks_value_loss_coef
    loss_config.kickstarter.teacher_lead_prob = 1.0
    loss_config.kickstarter.temperature = 1.0
    loss_config.kickstarter.teacher_uri = (
        "s3://softmax-public/policies/av.sliced.mb.11.22.110.ctrl/av.sliced.mb.11.22.110.ctrl:v9900.mpt"
    )

    trainer_updates: dict[str, Any] = {
        "losses": loss_config,
    }

    scheduler = SchedulerConfig(
        run_gates=[
            LossRunGate(loss_instance_name="ppo", phase="rollout", begin_at_step=1_500_000_000),
            LossRunGate(
                loss_instance_name="kickstarter",
                phase="rollout",
                end_at_step=1_500_000_000,
            ),
            LossRunGate(
                loss_instance_name="kickstarter",
                phase="train",
                end_at_step=1_500_000_000,
            ),
        ],
        rules=[
            HyperUpdateRule(
                loss_instance_name="kickstarter",
                attr_path="action_loss_coef",
                mode="progress",
                style="cosine",
                start_value=2.0,
                end_value=0.0,
                start_agent_step=800_000_000,
                end_agent_step=1_500_000_000,
            ),
            HyperUpdateRule(
                loss_instance_name="kickstarter",
                attr_path="value_loss_coef",
                mode="progress",
                style="cosine",
                start_value=1.0,
                end_value=0.0,
                start_agent_step=800_000_000,
                end_agent_step=1_500_000_000,
            ),
            HyperUpdateRule(
                loss_instance_name="kickstarter",
                attr_path="teacher_lead_prob",
                mode="progress",
                style="cosine",
                start_value=1.0,
                end_value=0.0,
                start_agent_step=800_000_000,
                end_agent_step=1_500_000_000,
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
    learning_rate: Optional[float] = None,
    value_loss_coef: Optional[float] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Train SmolLLM with optimized defaults for memory-constrained environments."""
    (
        policy_config,
        trainer_updates,
        env_updates,
        _num_agents,
        ks_act_loss_coef,
        default_ks_value_loss_coef,
    ) = _smollm_config(
        model_name,
        mem_len,
        learning_rate,
    )
    ks_value_loss_coef = float(value_loss_coef) if value_loss_coef is not None else default_ks_value_loss_coef
    kickstarter_trainer_updates, scheduler = _kickstarter_config(ks_act_loss_coef, ks_value_loss_coef)

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
