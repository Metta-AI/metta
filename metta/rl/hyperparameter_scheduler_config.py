from typing import Any, Dict, Optional

from metta.common.config import Config


class HyperparameterSchedulerConfig(Config):
    """
    Configuration for hyperparameter scheduling in RL training.
    """

    learning_rate_schedule: Optional[Dict[str, Any]] = None
    ppo_clip_schedule: Optional[Dict[str, Any]] = None
    ppo_ent_coef_schedule: Optional[Dict[str, Any]] = None
    ppo_vf_clip_schedule: Optional[Dict[str, Any]] = None
    ppo_l2_reg_loss_schedule: Optional[Dict[str, Any]] = None
    ppo_l2_init_loss_schedule: Optional[Dict[str, Any]] = None
