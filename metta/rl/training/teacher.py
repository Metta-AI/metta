from __future__ import annotations

from typing import Literal

from pydantic import Field

from metta.rl.trainer_config import TrainerConfig
from metta.rl.training.scheduler import HyperUpdateRule, LossRunGate
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from mettagrid.base_config import Config

TeacherMode = Literal[
    "sliced_cloner",
    "supervisor",
    "sliced_kickstarter",
    "kickstarter",
    "logit_kickstarter",
]

DEFAULT_TEACHER_STEPS = 1_000_000_000


class TeacherConfig(Config):
    """Shared knobs for enabling teacher/supervisor driven training phases."""

    policy_uri: str | None = None
    mode: TeacherMode = "sliced_cloner"
    steps: int | None = None
    # Teacher (led) and student slices should leave some remainder for PPO.
    # Match mainline BC defaults: start at 20% teacher-led, anneal to 0.
    led_proportion: float = Field(default=0.2, ge=0.0, le=1.0)
    student_proportion: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def enabled(self) -> bool:
        return self.policy_uri is not None


def apply_teacher_phase(
    *,
    trainer_cfg: TrainerConfig,
    training_env_cfg: TrainingEnvironmentConfig,
    scheduler_rules: list[HyperUpdateRule],
    scheduler_run_gates: list[LossRunGate],
    teacher_cfg: TeacherConfig,
    default_steps: int = DEFAULT_TEACHER_STEPS,
) -> None:
    """Enable and schedule the requested teacher loss."""

    if not teacher_cfg.enabled:
        return

    total_steps = teacher_cfg.steps if teacher_cfg.steps is not None else default_steps
    losses = trainer_cfg.losses

    if teacher_cfg.mode in {"sliced_cloner", "supervisor"}:
        _require_policy_uri(teacher_cfg)
        training_env_cfg.supervisor_policy_uri = teacher_cfg.policy_uri

    if teacher_cfg.mode == "sliced_cloner":
        _disable_ppo_until(total_steps, losses)
        _enable_loss(losses.sliced_scripted_cloner, teacher_cfg)
        _append_bc_run_gates(scheduler_run_gates, "sliced_scripted_cloner", total_steps, enable_ppo_after=total_steps)
        _append_teacher_share_rule(scheduler_rules, "sliced_scripted_cloner", teacher_cfg.led_proportion, total_steps)

    elif teacher_cfg.mode == "supervisor":
        sup = _enable_loss(losses.supervisor, teacher_cfg)
        _append_bc_run_gates(scheduler_run_gates, "supervisor", total_steps)
        _append_teacher_share_rule(scheduler_rules, "supervisor", teacher_cfg.led_proportion, total_steps)
        if total_steps:
            scheduler_rules.append(
                HyperUpdateRule(
                    loss_instance_name="supervisor",
                    attr_path="action_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=sup.action_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )

    elif teacher_cfg.mode == "sliced_kickstarter":
        _require_policy_uri(teacher_cfg)
        _disable_ppo_until(total_steps, losses)
        sk = _enable_loss(losses.sliced_kickstarter, teacher_cfg, needs_teacher_uri=True)
        _append_bc_run_gates(scheduler_run_gates, "sliced_kickstarter", total_steps, enable_ppo_after=total_steps)
        _append_teacher_share_rule(scheduler_rules, "sliced_kickstarter", teacher_cfg.led_proportion, total_steps)

    elif teacher_cfg.mode == "kickstarter":
        _require_policy_uri(teacher_cfg)
        ks = _enable_loss(losses.kickstarter, teacher_cfg, needs_teacher_uri=True, set_student=False)
        _append_bc_run_gates(scheduler_run_gates, "kickstarter", total_steps, enable_ppo_after=total_steps)
        _append_teacher_share_rule(scheduler_rules, "kickstarter", teacher_cfg.led_proportion, total_steps)

    elif teacher_cfg.mode == "logit_kickstarter":
        _require_policy_uri(teacher_cfg)
        lk = _enable_loss(losses.logit_kickstarter, teacher_cfg, needs_teacher_uri=True, set_student=False)
        _append_bc_run_gates(scheduler_run_gates, "logit_kickstarter", total_steps, enable_ppo_after=total_steps)
        _append_teacher_share_rule(scheduler_rules, "logit_kickstarter", teacher_cfg.led_proportion, total_steps)

    else:
        raise ValueError(f"Unsupported teacher mode '{teacher_cfg.mode}'")


def _require_policy_uri(cfg: TeacherConfig) -> None:
    if not cfg.policy_uri:
        raise ValueError(f"TeacherConfig.mode='{cfg.mode}' requires policy_uri to be set.")


def _append_bc_run_gates(
    run_gates: list[LossRunGate],
    *,
    loss_name: str,
    total_steps: int | None,
    enable_ppo_after: int | None = None,
) -> None:
    if total_steps:
        run_gates.append(
            LossRunGate(
                loss_instance_name=loss_name,
                phase="rollout",
                end_at_step=total_steps,
            )
        )
        run_gates.append(
            LossRunGate(
                loss_instance_name=loss_name,
                phase="train",
                end_at_step=total_steps,
            )
        )

    if enable_ppo_after:
        run_gates.append(
            LossRunGate(
                loss_instance_name="ppo_critic",
                phase="rollout",
                begin_at_step=enable_ppo_after,
            )
        )


def _disable_ppo_until(total_steps: int, losses) -> None:
    losses.ppo_critic.sample_enabled = False
    losses.ppo_critic.train_forward_enabled = False
    losses.ppo_critic.deferred_training_start_step = total_steps


def _enable_loss(
    loss_cfg,
    teacher_cfg: TeacherConfig,
    *,
    needs_teacher_uri: bool = False,
    set_student: bool = True,
):
    loss_cfg.enabled = True
    if hasattr(loss_cfg, "led_proportion"):
        loss_cfg.led_proportion = teacher_cfg.led_proportion
    if set_student and hasattr(loss_cfg, "student_proportion"):
        loss_cfg.student_proportion = teacher_cfg.student_proportion
    if needs_teacher_uri and hasattr(loss_cfg, "teacher_uri"):
        loss_cfg.teacher_uri = teacher_cfg.policy_uri
    return loss_cfg


def _append_teacher_share_rule(
    rules: list[HyperUpdateRule],
    *,
    loss_name: str,
    start_value: float,
    total_steps: int | None,
) -> None:
    if not total_steps or start_value <= 0.0:
        return
    rules.append(
        HyperUpdateRule(
            loss_instance_name=loss_name,
            attr_path="led_proportion",
            mode="progress",
            style="linear",
            start_value=start_value,
            end_value=0.0,
            start_agent_step=0,
            end_agent_step=total_steps,
        )
    )
