from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from metta.rl.trainer_config import TrainerConfig
from metta.rl.training.scheduler import RunGate, ScheduleRule
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from mettagrid.base_config import Config

TeacherMode = Literal[
    "sliced_cloner",
    "sliced_cloner_no_ppo",
    "supervisor",
    "sliced_kickstarter",
    "kickstarter",
    "logit_kickstarter",
    "eer_kickstarter",
    "eer_cloner",
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
    # Optional per-mode overrides applied to the selected teacher loss config.
    #
    # Example CLI usage:
    #   teacher.mode=eer_kickstarter teacher.policy_uri="s3://..." teacher.kwargs.r_lambda=0.25
    #
    # NOTE: This is applied inside apply_teacher_phase() so scheduled annealing rules
    # (e.g. r_lambda -> 0) start from the overridden value.
    kwargs: dict[str, Any] = Field(default_factory=dict)

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
    scheduler_rules: list[ScheduleRule],
    scheduler_run_gates: list[RunGate],
    teacher_cfg: TeacherConfig,
    default_steps: int = DEFAULT_TEACHER_STEPS,
) -> None:
    """Enable and schedule the requested teacher loss."""

    if not teacher_cfg.enabled:
        return

    total_steps = teacher_cfg.steps or default_steps
    nodes = trainer_cfg.nodes
    loss_cfg = _select_teacher_node_cfg(nodes=nodes, mode=teacher_cfg.mode)
    if loss_cfg is not None:
        _apply_teacher_kwargs(loss_cfg=loss_cfg, teacher_cfg=teacher_cfg)

    def _gate_loss(name: str, end_at_step: int = total_steps) -> None:
        if end_at_step:
            scheduler_run_gates.extend(
                [
                    RunGate(node_name=name, phase="rollout", end_at_step=end_at_step),
                    RunGate(node_name=name, phase="train", end_at_step=end_at_step),
                ]
            )

    def _gate_critic_after_teacher() -> None:
        if total_steps:
            scheduler_run_gates.append(RunGate(node_name="ppo_critic", phase="rollout", begin_at_step=total_steps))
            if trainer_cfg.nodes["quantile_ppo_critic"].enabled:
                scheduler_run_gates.append(
                    RunGate(node_name="quantile_ppo_critic", phase="rollout", begin_at_step=total_steps)
                )

    def _anneal(loss_name: str, attr_path: str, start_value: float) -> None:
        if total_steps and start_value > 0.0:
            scheduler_rules.append(
                ScheduleRule(
                    target_path=f"nodes.{loss_name}.{attr_path}",
                    mode="progress",
                    style="linear",
                    start_value=start_value,
                    end_value=0.0,
                    start_agent_step=0,
                    end_agent_step=total_steps,
                )
            )

    if teacher_cfg.mode in {"sliced_cloner", "sliced_cloner_no_ppo", "supervisor", "eer_cloner"}:
        _require_policy_uri(teacher_cfg)
        training_env_cfg.supervisor_policy_uri = teacher_cfg.policy_uri

    if teacher_cfg.mode == "sliced_cloner":
        slicer = nodes["sliced_scripted_cloner"]
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
        slicer = nodes["sliced_scripted_cloner"]
        slicer.enabled = True
        slicer.teacher_led_proportion = teacher_cfg.teacher_led_proportion
        slicer.student_led_proportion = teacher_cfg.student_led_proportion

        _gate_loss("sliced_scripted_cloner")
        _gate_critic_after_teacher()
        nodes["ppo_critic"].vf_coef = 0.0
        nodes["ppo_actor"].enabled = False
        _anneal(
            "sliced_scripted_cloner", attr_path="teacher_led_proportion", start_value=teacher_cfg.teacher_led_proportion
        )
        _anneal(
            "sliced_scripted_cloner",
            attr_path="student_led_proportion",
            start_value=teacher_cfg.student_led_proportion,
        )

    elif teacher_cfg.mode == "supervisor":
        supervisor = nodes["supervisor"]
        supervisor.enabled = True
        supervisor.teacher_led_proportion = teacher_cfg.teacher_led_proportion

        # Legacy BC behavior: stay in pure-supervisor mode for the whole run.
        # Do not gate off the supervisor or re-enable PPO later.
        nodes["ppo_actor"].enabled = False
        nodes["ppo_critic"].enabled = False
        nodes["quantile_ppo_critic"].enabled = False
        scheduler_run_gates.clear()
        scheduler_rules.clear()

    elif teacher_cfg.mode == "sliced_kickstarter":
        _require_policy_uri(teacher_cfg)
        sliced_kick = nodes["sliced_kickstarter"]
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

    elif teacher_cfg.mode == "eer_kickstarter":
        _require_policy_uri(teacher_cfg)
        eer_kick = nodes["eer_kickstarter"]
        eer_kick.enabled = True
        eer_kick.teacher_uri = teacher_cfg.policy_uri

        _gate_loss("eer_kickstarter")
        _gate_critic_after_teacher()
        if total_steps:
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.eer_kickstarter.action_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=eer_kick.action_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.eer_kickstarter.value_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=eer_kick.value_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.eer_kickstarter.r_lambda",
                    mode="progress",
                    style="linear",
                    start_value=eer_kick.r_lambda,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )
    elif teacher_cfg.mode == "kickstarter":
        _require_policy_uri(teacher_cfg)
        ks = nodes["kickstarter"]
        ks.enabled = True
        ks.teacher_uri = teacher_cfg.policy_uri
        ks.teacher_led_proportion = teacher_cfg.teacher_led_proportion

        _gate_loss("kickstarter")
        _gate_critic_after_teacher()
        _anneal("kickstarter", attr_path="teacher_led_proportion", start_value=teacher_cfg.teacher_led_proportion)
        if total_steps:
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.kickstarter.action_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=ks.action_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.kickstarter.value_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=ks.value_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )

    elif teacher_cfg.mode == "logit_kickstarter":
        _require_policy_uri(teacher_cfg)
        logit = nodes["logit_kickstarter"]
        logit.enabled = True
        logit.teacher_uri = teacher_cfg.policy_uri
        logit.teacher_led_proportion = teacher_cfg.teacher_led_proportion

        _gate_loss("logit_kickstarter")
        _gate_critic_after_teacher()
        _anneal("logit_kickstarter", attr_path="teacher_led_proportion", start_value=teacher_cfg.teacher_led_proportion)
        if total_steps:
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.logit_kickstarter.action_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=logit.action_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.logit_kickstarter.value_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=logit.value_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )

    elif teacher_cfg.mode == "eer_cloner":
        _require_policy_uri(teacher_cfg)
        eer_cl = nodes["eer_cloner"]
        eer_cl.enabled = True

        _gate_loss("eer_cloner")
        _gate_critic_after_teacher()
        if total_steps:
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.eer_cloner.action_loss_coef",
                    mode="progress",
                    style="linear",
                    start_value=eer_cl.action_loss_coef,
                    end_value=0.0,
                    start_agent_step=total_steps // 2,
                    end_agent_step=total_steps,
                )
            )
            scheduler_rules.append(
                ScheduleRule(
                    target_path="nodes.eer_cloner.r_lambda",
                    mode="progress",
                    style="linear",
                    start_value=eer_cl.r_lambda,
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


def _apply_teacher_kwargs(*, loss_cfg: Config, teacher_cfg: TeacherConfig) -> None:
    """Apply teacher.kwargs overrides onto a concrete loss config.

    This is intentionally applied during apply_teacher_phase() so any scheduled
    ScheduleRules use the overridden start values.
    """

    if not teacher_cfg.kwargs:
        return
    for key, value in teacher_cfg.kwargs.items():
        try:
            loss_cfg.override(key, value)
        except Exception as e:  # noqa: BLE001 - surface as a user-facing config error
            raise ValueError(f"Invalid teacher.kwargs override for mode='{teacher_cfg.mode}': {key}={value!r}") from e


def _select_teacher_node_cfg(*, nodes: dict[str, Any], mode: TeacherMode) -> Config | None:
    if mode in {"sliced_cloner", "sliced_cloner_no_ppo"}:
        return nodes.get("sliced_scripted_cloner")
    if mode == "supervisor":
        return nodes.get("supervisor")
    if mode == "sliced_kickstarter":
        return nodes.get("sliced_kickstarter")
    if mode == "eer_kickstarter":
        return nodes.get("eer_kickstarter")
    if mode == "kickstarter":
        return nodes.get("kickstarter")
    if mode == "logit_kickstarter":
        return nodes.get("logit_kickstarter")
    if mode == "eer_cloner":
        return nodes.get("eer_cloner")
    return None
