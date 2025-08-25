from typing import Any, Dict, Optional

from metta.common.config import Config


class HyperparameterSchedulerConfig(Config):
    """
    Configuration for hyperparameter scheduling in RL training.

    Provides default schedules that match the pre-dehydration reference configuration
    from /tmp/metta-reference/configs/trainer/trainer.yaml lines 85-113.
    """

    learning_rate_schedule: Optional[Dict[str, Any]] = {
        "_target_": "metta.rl.hyperparameter_scheduler.CosineSchedule",
        "min_value": 0.00003,
        # initial_value will be taken from optimizer.learning_rate (0.000457)
    }

    ppo_clip_schedule: Optional[Dict[str, Any]] = {
        "_target_": "metta.rl.hyperparameter_scheduler.LogarithmicSchedule",
        "min_value": 0.05,
        "decay_rate": 0.1,
        # initial_value will be taken from ppo.clip_coef (0.1)
    }

    ppo_ent_coef_schedule: Optional[Dict[str, Any]] = {
        "_target_": "metta.rl.hyperparameter_scheduler.LinearSchedule",
        "min_value": 0.0,
        # initial_value will be taken from ppo.ent_coef (0.0021)
    }

    ppo_vf_clip_schedule: Optional[Dict[str, Any]] = {
        "_target_": "metta.rl.hyperparameter_scheduler.LinearSchedule",
        "min_value": 0.05,
        # initial_value will be taken from ppo.vf_clip_coef (0.1)
    }

    ppo_l2_reg_loss_schedule: Optional[Dict[str, Any]] = {
        "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
        # initial_value will be taken from ppo.l2_reg_loss_coef (0)
    }

    ppo_l2_init_loss_schedule: Optional[Dict[str, Any]] = {
        "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
        # initial_value will be taken from ppo.l2_init_loss_coef (0)
    }
