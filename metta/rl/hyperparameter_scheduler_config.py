from dataclasses import dataclass
from typing import Optional

from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra


@dataclass
class SchedulerConfig:
    initial_value: float
    _target_: str = Field(..., alias="_target_")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    decay_rate: Optional[float] = None

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True


class HyperparameterSchedulerConfig(BaseModelWithForbidExtra):
    """
    Configuration for hyperparameter scheduling in RL training.
    """

    learning_rate_schedule: Optional[SchedulerConfig] = None  # Use Any instead of SchedulerConfig
    ppo_clip_schedule: Optional[SchedulerConfig] = None
    ppo_ent_coef_schedule: Optional[SchedulerConfig] = None
    ppo_vf_clip_schedule: Optional[SchedulerConfig] = None
    ppo_l2_reg_loss_schedule: Optional[SchedulerConfig] = None
    ppo_l2_init_loss_schedule: Optional[SchedulerConfig] = None
