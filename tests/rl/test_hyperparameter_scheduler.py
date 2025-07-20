import logging

from omegaconf import DictConfig

from metta.rl.hyperparameter_scheduler import (
    ConstantSchedule,
    CosineSchedule,
    ExponentialSchedule,
    HyperparameterScheduler,
    LinearSchedule,
    LogarithmicSchedule,
)


class TestSchedules:
    def test_constant_schedule(self):
        schedule = ConstantSchedule(initial_value=0.001)
        assert schedule(0.0) == 0.001
        assert schedule(0.5) == 0.001
        assert schedule(1.0) == 0.001

    def test_linear_schedule(self):
        schedule = LinearSchedule(initial_value=1.0, min_value=0.0)
        assert schedule(0.0) == 1.0
        assert abs(schedule(0.5) - 0.5) < 1e-6
        assert schedule(1.0) == 0.0

    def test_cosine_schedule(self):
        schedule = CosineSchedule(initial_value=1.0, min_value=0.0)
        assert schedule(0.0) == 1.0
        assert abs(schedule(0.5) - 0.5) < 1e-6  # Cosine at pi/2
        assert abs(schedule(1.0) - 0.0) < 1e-6

    def test_exponential_schedule(self):
        schedule = ExponentialSchedule(initial_value=1.0, min_value=0.1, decay_rate=0.5)
        assert schedule(0.0) == 1.0
        # After half the training, value should be initial_value * decay_rate^0.5
        assert abs(schedule(0.5) - (1.0 * (0.5**0.5))) < 1e-6
        # Should not go below min_value
        assert schedule(10.0) >= 0.1

    def test_logarithmic_schedule(self):
        schedule = LogarithmicSchedule(initial_value=1.0, min_value=0.0, decay_rate=0.1)
        assert schedule(0.0) == 1.0
        # Should decrease gradually
        assert schedule(0.5) < 1.0
        assert schedule(0.5) > 0.0
        assert abs(schedule(1.0) - 0.0) < 1e-6


class MockTrainer:
    """Mock trainer for testing hyperparameter scheduler."""

    def __init__(self):
        self.optimizer = type("Optimizer", (), {"param_groups": [{"lr": 0.001}]})()
        self.trainer_cfg = DictConfig(
            {
                "ppo": {
                    "clip_coef": 0.2,
                    "vf_clip_coef": 0.1,
                    "ent_coef": 0.01,
                    "l2_reg_loss_coef": 0.0,
                    "l2_init_loss_coef": 0.0,
                }
            }
        )


class TestHyperparameterScheduler:
    def test_scheduler_integration(self):
        # Create mock trainer
        trainer = MockTrainer()

        # Create scheduler config
        scheduler_cfg = DictConfig(
            {
                "hyperparameter_scheduler": {
                    "learning_rate_schedule": {
                        "_target_": "metta.rl.hyperparameter_scheduler.LinearSchedule",
                        "initial_value": 0.001,
                        "min_value": 0.0001,
                    },
                    "ppo_clip_schedule": {
                        "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
                        "initial_value": 0.2,
                    },
                    "ppo_vf_clip_schedule": {
                        "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
                        "initial_value": 0.1,
                    },
                    "ppo_ent_coef_schedule": {
                        "_target_": "metta.rl.hyperparameter_scheduler.LinearSchedule",
                        "initial_value": 0.01,
                        "min_value": 0.0,
                    },
                    "ppo_l2_reg_loss_schedule": {
                        "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
                        "initial_value": 0.0,
                    },
                    "ppo_l2_init_loss_schedule": {
                        "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
                        "initial_value": 0.0,
                    },
                }
            }
        )

        # Add scheduler config to trainer config
        trainer.trainer_cfg.hyperparameter_scheduler = scheduler_cfg.hyperparameter_scheduler

        # Create scheduler
        total_timesteps = 1000
        scheduler = HyperparameterScheduler(trainer.trainer_cfg, trainer, total_timesteps, logging)

        # Test initial values
        assert trainer.optimizer.param_groups[0]["lr"] == 0.001
        assert trainer.trainer_cfg.ppo.clip_coef == 0.2
        assert trainer.trainer_cfg.ppo.ent_coef == 0.01

        # Step halfway through training
        scheduler.step(500)

        # Check values have been updated appropriately
        assert abs(trainer.optimizer.param_groups[0]["lr"] - 0.00055) < 1e-6  # Linear from 0.001 to 0.0001
        assert trainer.trainer_cfg.ppo.clip_coef == 0.2  # Constant
        assert abs(trainer.trainer_cfg.ppo.ent_coef - 0.005) < 1e-6  # Linear from 0.01 to 0.0

        # Step to end of training
        scheduler.step(1000)

        # Check final values
        assert abs(trainer.optimizer.param_groups[0]["lr"] - 0.0001) < 1e-6
        assert trainer.trainer_cfg.ppo.clip_coef == 0.2
        assert abs(trainer.trainer_cfg.ppo.ent_coef - 0.0) < 1e-6
