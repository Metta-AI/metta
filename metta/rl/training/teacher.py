from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from metta.rl.trainer_config import TrainerConfig
from metta.rl.training.scheduler import HyperUpdateRule, LossRunGate
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from mettagrid.base_config import Config

TeacherMode = Literal[
    "sliced_cloner",
    "sliced_cloner_no_ppo",
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

    @model_validator(mode="after")
    def validate_proportions(self) -> "TeacherConfig":
        total = self.teacher_led_proportion + self.student_led_proportion
        if total > 1.0:
            raise ValueError(
                f"teacher_led_proportion ({self.teacher_led_proportion}) + "
                f"student_led_proportion ({self.student_led_proportion}) must be <= 1.0"
            )
        return self


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

    total_steps = teacher_cfg.steps or default_steps
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

    def _anneal(loss_name: str, attr_path: str, start_value: float) -> None:
        if total_steps and start_value > 0.0:
            scheduler_rules.append(
                HyperUpdateRule(
                    loss_instance_name=loss_name,
                    attr_path=attr_path,
                    mode="progress",
                    style="linear",
                    start_value=start_value,
                    end_value=0.0,
                    start_agent_step=0,
                    end_agent_step=total_steps,
                )
            )

    if teacher_cfg.mode in {"sliced_cloner", "supervisor"}:
        _require_policy_uri(teacher_cfg)
        training_env_cfg.supervisor_policy_uri = teacher_cfg.policy_uri

    if teacher_cfg.mode == "sliced_cloner":
        slicer = losses.sliced_scripted_cloner
        slicer.enabled = True
        slicer.teacher_led_proportion = teacher_cfg.teacher_led_proportion
        slicer.student_led_proportion = teacher_cfg.student_led_proportion

        _gate_loss("sliced_scripted_cloner")
        _gate_critic_after_teacher()
        _anneal(
            "sliced_scripted_cloner", attr_path="teacher_led_proportion", start_value=teacher_cfg.teacher_led_proportion
        )
        _anneal(
            "sliced_scripted_cloner",
            attr_path="student_led_proportion",
            start_value=teacher_cfg.student_led_proportion,
        )

    elif teacher_cfg.mode == "sliced_cloner_no_ppo":
        """This will anneal teacher-led and student-led loss proportions, thereby leaving the PPO critic proportions to
        grow. PPO critic won't update weights because vf_coef is 0.0. PPO actor is disabled."""
        slicer = losses.sliced_scripted_cloner
        slicer.enabled = True
        slicer.teacher_led_proportion = teacher_cfg.teacher_led_proportion
        slicer.student_led_proportion = teacher_cfg.student_led_proportion

        _gate_loss("sliced_scripted_cloner")
        _gate_critic_after_teacher()
        losses.ppo_critic.vf_coef = 0.0
        losses.ppo_actor.enabled = False
        _anneal(
            "sliced_scripted_cloner", attr_path="teacher_led_proportion", start_value=teacher_cfg.teacher_led_proportion
        )
        _anneal(
            "sliced_scripted_cloner",
            attr_path="student_led_proportion",
            start_value=teacher_cfg.student_led_proportion,
        )

    elif teacher_cfg.mode == "supervisor":
        supervisor = losses.supervisor
        supervisor.enabled = True
        supervisor.teacher_led_proportion = teacher_cfg.teacher_led_proportion

        _gate_loss("supervisor")
        _gate_critic_after_teacher()
        _anneal("supervisor", attr_path="teacher_led_proportion", start_value=teacher_cfg.teacher_led_proportion)
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
        sliced_kick = losses.sliced_kickstarter
        sliced_kick.enabled = True
        sliced_kick.teacher_uri = teacher_cfg.policy_uri
        sliced_kick.teacher_led_proportion = teacher_cfg.teacher_led_proportion
        sliced_kick.student_led_proportion = teacher_cfg.student_led_proportion

        _gate_loss("sliced_kickstarter")
        _gate_critic_after_teacher()
        _anneal(
            "sliced_kickstarter", attr_path="teacher_led_proportion", start_value=teacher_cfg.teacher_led_proportion
        )
        _anneal(
            "sliced_kickstarter",
            attr_path="student_led_proportion",
            start_value=teacher_cfg.student_led_proportion,
        )

    elif teacher_cfg.mode == "kickstarter":
        _require_policy_uri(teacher_cfg)
        ks = losses.kickstarter
        ks.enabled = True
        ks.teacher_uri = teacher_cfg.policy_uri
        ks.teacher_led_proportion = teacher_cfg.teacher_led_proportion

        _gate_loss("kickstarter")
        _gate_critic_after_teacher()
        _anneal("kickstarter", attr_path="teacher_led_proportion", start_value=teacher_cfg.teacher_led_proportion)

    elif teacher_cfg.mode == "logit_kickstarter":
        _require_policy_uri(teacher_cfg)
        logit = losses.logit_kickstarter
        logit.enabled = True
        logit.teacher_uri = teacher_cfg.policy_uri

        _gate_loss("logit_kickstarter")
        _gate_critic_after_teacher()
        if total_steps:
            scheduler_rules.append(
                HyperUpdateRule(
                    loss_instance_name="logit_kickstarter",
                    attr_path="action_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=logit.action_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )

    else:
        raise ValueError(f"Unsupported teacher mode '{teacher_cfg.mode}'")


def _require_policy_uri(cfg: TeacherConfig) -> None:
    if not cfg.policy_uri:
        raise ValueError(f"TeacherConfig.mode='{cfg.mode}' requires policy_uri to be set.")
