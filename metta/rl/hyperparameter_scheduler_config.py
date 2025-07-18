from typing import Optional

from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra


class SchedulerConfig(BaseModelWithForbidExtra):
    initial_value: float
    target_: str = Field(default="metta.rl.hyperparameter_scheduler.ConstantSchedule", alias="_target_")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    decay_rate: Optional[float] = None


class HyperparameterSchedulerConfig(BaseModelWithForbidExtra):
    """
    Configuration for hyperparameter scheduling in RL training.
    """

    learning_rate_schedule: Optional[SchedulerConfig] = None
    ppo_clip_schedule: Optional[SchedulerConfig] = None
    ppo_ent_coef_schedule: Optional[SchedulerConfig] = None
    ppo_vf_clip_schedule: Optional[SchedulerConfig] = None
    ppo_l2_reg_loss_schedule: Optional[SchedulerConfig] = None
    ppo_l2_init_loss_schedule: Optional[SchedulerConfig] = None
