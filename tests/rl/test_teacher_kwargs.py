"""Tests for teacher kwargs overrides in apply_teacher_phase()."""

from metta.rl.trainer_config import TrainerConfig
from metta.rl.training.scheduler import HyperUpdateRule, LossRunGate
from metta.rl.training.teacher import TeacherConfig, apply_teacher_phase
from metta.rl.training.training_environment import TrainingEnvironmentConfig


def test_teacher_kwargs_overrides_eer_kickstarter_r_lambda_and_scheduler_start_value() -> None:
    trainer_cfg = TrainerConfig()
    training_env_cfg = TrainingEnvironmentConfig()
    scheduler_rules: list[HyperUpdateRule] = []
    scheduler_run_gates: list[LossRunGate] = []

    teacher_cfg = TeacherConfig(
        policy_uri="s3://softmax-public/policies/example/teacher:v0.mpt",
        mode="eer_kickstarter",
        kwargs={"r_lambda": 0.25},
    )

    apply_teacher_phase(
        trainer_cfg=trainer_cfg,
        training_env_cfg=training_env_cfg,
        scheduler_rules=scheduler_rules,
        scheduler_run_gates=scheduler_run_gates,
        teacher_cfg=teacher_cfg,
        default_steps=100,
    )

    assert trainer_cfg.losses.eer_kickstarter.enabled is True
    assert trainer_cfg.losses.eer_kickstarter.teacher_uri == teacher_cfg.policy_uri
    assert trainer_cfg.losses.eer_kickstarter.r_lambda == 0.25

    rule = next(r for r in scheduler_rules if r.loss_instance_name == "eer_kickstarter" and r.attr_path == "r_lambda")
    assert rule.start_value == 0.25
