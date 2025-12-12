from __future__ import annotations

from typing import Literal

from pydantic import Field

from metta.rl.slot import PolicySlotConfig
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
    # Optional slot wiring: teacher/student slot ids used during teacher-led phases.
    teacher_slot_id: str = Field(default="teacher")
    student_slot_id: str = Field(default="main")

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

    def _ensure_teacher_slot() -> None:
        if not teacher_cfg.policy_uri:
            return
        slots = list(trainer_cfg.policy_slots or [])
        if any(slot.id == teacher_cfg.teacher_slot_id for slot in slots):
            trainer_cfg.policy_slots = slots
            return
        slots.append(
            PolicySlotConfig(
                id=teacher_cfg.teacher_slot_id,
                policy_uri=teacher_cfg.policy_uri,
                class_path=None,
                policy_kwargs={},
                trainable=False,
                loss_profile=None,
                use_trainer_policy=False,
            )
        )
        trainer_cfg.policy_slots = slots

    def _wire_slot_ids(loss_cfg: object) -> None:
        if hasattr(loss_cfg, "teacher_slot_id"):
            setattr(loss_cfg, "teacher_slot_id", teacher_cfg.teacher_slot_id)
        if hasattr(loss_cfg, "student_slot_id"):
            setattr(loss_cfg, "student_slot_id", teacher_cfg.student_slot_id)

    total_steps = teacher_cfg.steps if teacher_cfg.steps is not None else default_steps
    losses = trainer_cfg.losses

    def _gate_loss(name: str) -> None:
        if total_steps:
            scheduler_run_gates.extend(
                [
                    LossRunGate(loss_instance_name=name, phase="rollout", end_at_step=total_steps),
                    LossRunGate(loss_instance_name=name, phase="train", end_at_step=total_steps),
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
                    attr_path="led_proportion",
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
        _ensure_teacher_slot()

    if teacher_cfg.mode == "sliced_cloner":
        losses.ppo_critic.sample_enabled = True
        losses.ppo_critic.train_forward_enabled = True
        losses.ppo_critic.rollout_forward_enabled = True
        losses.ppo_critic.deferred_training_start_step = None

        slicer = losses.sliced_scripted_cloner
        slicer.enabled = True
        slicer.led_proportion = teacher_cfg.led_proportion
        slicer.student_proportion = teacher_cfg.student_proportion
        _wire_slot_ids(slicer)

        _gate_loss("sliced_scripted_cloner")
        _anneal_led("sliced_scripted_cloner", teacher_cfg.led_proportion)

    elif teacher_cfg.mode == "supervisor":
        supervisor = losses.supervisor
        supervisor.enabled = True
        supervisor.led_proportion = teacher_cfg.led_proportion

        _gate_loss("supervisor")
        _anneal_led("supervisor", teacher_cfg.led_proportion)
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
        sliced_kick.led_proportion = teacher_cfg.led_proportion
        sliced_kick.student_proportion = teacher_cfg.student_proportion
        _ensure_teacher_slot()
        _wire_slot_ids(sliced_kick)

        _gate_loss("sliced_kickstarter")
        _gate_critic_after_teacher()
        _anneal_led("sliced_kickstarter", teacher_cfg.led_proportion)

    elif teacher_cfg.mode == "kickstarter":
        _require_policy_uri(teacher_cfg)
        ks = losses.kickstarter
        ks.enabled = True
        ks.teacher_uri = teacher_cfg.policy_uri
        ks.led_proportion = teacher_cfg.led_proportion
        _ensure_teacher_slot()

        _gate_loss("kickstarter")
        _gate_critic_after_teacher()
        _anneal_led("kickstarter", teacher_cfg.led_proportion)

    elif teacher_cfg.mode == "logit_kickstarter":
        _require_policy_uri(teacher_cfg)
        logit = losses.logit_kickstarter
        logit.enabled = True
        logit.teacher_uri = teacher_cfg.policy_uri
        logit.led_proportion = teacher_cfg.led_proportion
        _ensure_teacher_slot()

        _gate_loss("logit_kickstarter")
        _gate_critic_after_teacher()
        _anneal_led("logit_kickstarter", teacher_cfg.led_proportion)

    else:
        raise ValueError(f"Unsupported teacher mode '{teacher_cfg.mode}'")


def _require_policy_uri(cfg: TeacherConfig) -> None:
    if not cfg.policy_uri:
        raise ValueError(f"TeacherConfig.mode='{cfg.mode}' requires policy_uri to be set.")
