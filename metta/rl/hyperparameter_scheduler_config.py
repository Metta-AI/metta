from typing import Any, Dict, Optional

from metta.common.util.typed_config import BaseModelWithForbidExtra


class HyperparameterSchedulerConfig(BaseModelWithForbidExtra):
    """
    Configuration for hyperparameter scheduling in RL training.

    Each schedule field should be a dict that can be passed to hydra.utils.instantiate
    with a _target_ field specifying the schedule class.
    """

    learning_rate_schedule: Optional[Dict[str, Any]] = None
    ppo_clip_schedule: Optional[Dict[str, Any]] = None
    ppo_ent_coef_schedule: Optional[Dict[str, Any]] = None
    ppo_vf_clip_schedule: Optional[Dict[str, Any]] = None
    ppo_l2_reg_loss_schedule: Optional[Dict[str, Any]] = None
    ppo_l2_init_loss_schedule: Optional[Dict[str, Any]] = None
