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
    teacher_led_proportion: float = Field(default=0.2, ge=0.0, le=1.0)
    student_led_proportion: float = Field(default=0.0, ge=0.0, le=1.0)

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

    def _gate_loss(name: str, end_at_step: int | None = total_steps) -> None:
        if end_at_step:
            scheduler_run_gates.extend(
                [
                    LossRunGate(loss_instance_name=name, phase="rollout", end_at_step=end_at_step),
                    LossRunGate(loss_instance_name=name, phase="train", end_at_step=end_at_step),
                ]
            )

    def _gate_critic_after_teacher() -> None:
        if total_steps:
            scheduler_run_gates.append(
                LossRunGate(loss_instance_name="ppo_critic", phase="rollout", begin_at_step=total_steps)
            )

    def _anneal_led(loss_name: str, start_value: float) -> None:
        if total_steps and start_value > 0.0:
            scheduler_rules.append(
                HyperUpdateRule(
                    loss_instance_name=loss_name,
                    attr_path="teacher_led_proportion",
                    mode="progress",
                    style="linear",
                    start_value=start_value,
                    end_value=0.0,
                    start_agent_step=0,
                    end_agent_step=total_steps,
                )
            )

    def _grow_student(loss_name: str, start_value: float, teacher_start: float) -> None:
        """Ramp student proportion up to (teacher_start + student_start), not beyond 1.0."""
        if not total_steps or start_value <= 0.0:
            return
        target = min(1.0, start_value + teacher_start)
        if target <= start_value:
            return
        scheduler_rules.append(
            HyperUpdateRule(
                loss_instance_name=loss_name,
                attr_path="student_led_proportion",
                mode="progress",
                style="linear",
                start_value=start_value,
                end_value=target,
                start_agent_step=0,
                end_agent_step=total_steps,
            )
        )

    if teacher_cfg.mode in {"sliced_cloner", "supervisor"}:
        _require_policy_uri(teacher_cfg)
        training_env_cfg.supervisor_policy_uri = teacher_cfg.policy_uri

    if teacher_cfg.mode == "sliced_cloner":
        # If teacher + student consume the whole batch, skip PPO entirely.
        total_led = teacher_cfg.teacher_led_proportion + teacher_cfg.student_led_proportion
        disable_ppo = total_led >= 1.0
        keep_cloner_after_teacher = teacher_cfg.student_led_proportion > 0.0
        gate_end_step = None if keep_cloner_after_teacher else total_steps

        losses.ppo_critic.sample_enabled = True
        losses.ppo_critic.train_forward_enabled = True
        losses.ppo_critic.rollout_forward_enabled = True
        losses.ppo_critic.deferred_training_start_step = None

        slicer = losses.sliced_scripted_cloner
        slicer.enabled = True
        slicer.teacher_led_proportion = teacher_cfg.teacher_led_proportion
        slicer.student_led_proportion = teacher_cfg.student_led_proportion

        _gate_loss("sliced_scripted_cloner", end_at_step=gate_end_step)
        _anneal_led("sliced_scripted_cloner", teacher_cfg.teacher_led_proportion)
        _grow_student(
            "sliced_scripted_cloner",
            teacher_cfg.student_led_proportion,
            teacher_cfg.teacher_led_proportion,
        )

        if disable_ppo:
            for loss_name in ("ppo", "ppo_actor", "ppo_critic", "quantile_ppo_critic"):
                if hasattr(losses, loss_name):
                    getattr(losses, loss_name).enabled = False

    elif teacher_cfg.mode == "supervisor":
        supervisor = losses.supervisor
        supervisor.enabled = True
        supervisor.teacher_led_proportion = teacher_cfg.teacher_led_proportion

        _gate_loss("supervisor")
        _anneal_led("supervisor", teacher_cfg.teacher_led_proportion)
        if total_steps:
            scheduler_rules.append(
                HyperUpdateRule(
                    loss_instance_name="supervisor",
                    attr_path="action_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=supervisor.action_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )

    elif teacher_cfg.mode == "sliced_kickstarter":
        _require_policy_uri(teacher_cfg)
        losses.ppo_critic.sample_enabled = False
        losses.ppo_critic.train_forward_enabled = False
        losses.ppo_critic.deferred_training_start_step = total_steps

        sliced_kick = losses.sliced_kickstarter
        sliced_kick.enabled = True
        sliced_kick.teacher_uri = teacher_cfg.policy_uri
        sliced_kick.teacher_led_proportion = teacher_cfg.teacher_led_proportion
        sliced_kick.student_led_proportion = teacher_cfg.student_led_proportion

        _gate_loss("sliced_kickstarter")
        _gate_critic_after_teacher()
        _anneal_led("sliced_kickstarter", teacher_cfg.teacher_led_proportion)

    elif teacher_cfg.mode == "kickstarter":
        _require_policy_uri(teacher_cfg)
        ks = losses.kickstarter
        ks.enabled = True
        ks.teacher_uri = teacher_cfg.policy_uri
        ks.teacher_led_proportion = teacher_cfg.teacher_led_proportion

        _gate_loss("kickstarter")
        _gate_critic_after_teacher()
        _anneal_led("kickstarter", teacher_cfg.teacher_led_proportion)

    elif teacher_cfg.mode == "logit_kickstarter":
        _require_policy_uri(teacher_cfg)
        logit = losses.logit_kickstarter
        logit.enabled = True
        logit.teacher_uri = teacher_cfg.policy_uri
        logit.teacher_led_proportion = teacher_cfg.teacher_led_proportion

        _gate_loss("logit_kickstarter")
        _gate_critic_after_teacher()
        _anneal_led("logit_kickstarter", teacher_cfg.teacher_led_proportion)

    else:
        raise ValueError(f"Unsupported teacher mode '{teacher_cfg.mode}'")


def _require_policy_uri(cfg: TeacherConfig) -> None:
    if not cfg.policy_uri:
        raise ValueError(f"TeacherConfig.mode='{cfg.mode}' requires policy_uri to be set.")
