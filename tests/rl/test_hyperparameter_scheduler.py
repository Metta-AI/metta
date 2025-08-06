import logging

import torch
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


# Integration tests for the HyperparameterScheduler no longer rely on a mock trainer.
# Instead, they construct the minimal objects actually consumed by the scheduler:
# 1. a DictConfig with the relevant PPO and optimizer fields
# 2. a real `torch.optim.Optimizer` instance whose `param_groups` will be updated.


class TestHyperparameterScheduler:
    def test_scheduler_integration(self):
        # Build a minimal trainer configuration
        trainer_cfg = DictConfig(
            {
                "ppo": {
                    "clip_coef": 0.2,
                    "vf_clip_coef": 0.1,
                    "ent_coef": 0.01,
                    "l2_reg_loss_coef": 0.0,
                    "l2_init_loss_coef": 0.0,
                },
                "optimizer": {
                    "learning_rate": 0.001,
                },
            }
        )

        # Scheduler sub-configuration
        trainer_cfg.hyperparameter_scheduler = DictConfig(
            {
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
        )

        # Simple model and optimizer for testing updates to learning rate
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_cfg.optimizer.learning_rate)

        total_timesteps = 1000
        scheduler = HyperparameterScheduler.from_trainer_config(trainer_cfg, optimizer, total_timesteps, logging)

        # Create update callbacks to update trainer_cfg
        update_callbacks = {
            "ppo_clip_coef": lambda v: setattr(trainer_cfg.ppo, "clip_coef", v),
            "ppo_vf_clip_coef": lambda v: setattr(trainer_cfg.ppo, "vf_clip_coef", v),
            "ppo_ent_coef": lambda v: setattr(trainer_cfg.ppo, "ent_coef", v),
            "ppo_l2_reg_loss_coef": lambda v: setattr(trainer_cfg.ppo, "l2_reg_loss_coef", v),
            "ppo_l2_init_loss_coef": lambda v: setattr(trainer_cfg.ppo, "l2_init_loss_coef", v),
        }

        # Halfway through training
        scheduler.step(500, update_callbacks)

        assert abs(optimizer.param_groups[0]["lr"] - 0.00055) < 1e-6  # Linear schedule
        assert trainer_cfg.ppo.clip_coef == 0.2  # Constant schedule
        assert abs(trainer_cfg.ppo.ent_coef - 0.005) < 1e-6  # Linear schedule

        # End of training
        scheduler.step(1000, update_callbacks)

        assert abs(optimizer.param_groups[0]["lr"] - 0.0001) < 1e-6
        assert trainer_cfg.ppo.clip_coef == 0.2
        assert abs(trainer_cfg.ppo.ent_coef - 0.0) < 1e-6
