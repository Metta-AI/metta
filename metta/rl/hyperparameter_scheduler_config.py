from typing import Optional, Union

from metta.common.config import Config


class ConstantScheduleConfig(Config):
    type: str = "constant"
    initial_value: float


class LinearScheduleConfig(Config):
    type: str = "linear"
    initial_value: float
    min_value: Optional[float] = None


class CosineScheduleConfig(Config):
    type: str = "cosine"
    initial_value: float
    min_value: Optional[float] = None


class ExponentialScheduleConfig(Config):
    type: str = "exponential"
    initial_value: float
    decay_rate: float = 0.95
    min_value: Optional[float] = None


class LogarithmicScheduleConfig(Config):
    type: str = "logarithmic"
    initial_value: float
    min_value: Optional[float] = None
    decay_rate: float = 0.1


# Union type for all schedule configurations
ScheduleConfig = Union[
    ConstantScheduleConfig,
    LinearScheduleConfig,
    CosineScheduleConfig,
    ExponentialScheduleConfig,
    LogarithmicScheduleConfig,
]


class HyperparameterSchedulerConfig(Config):
    learning_rate_schedule: Optional[ScheduleConfig] = None
    ppo_clip_schedule: Optional[ScheduleConfig] = None
    ppo_ent_coef_schedule: Optional[ScheduleConfig] = None
    ppo_vf_clip_schedule: Optional[ScheduleConfig] = None
    ppo_l2_reg_loss_schedule: Optional[ScheduleConfig] = None
    ppo_l2_init_loss_schedule: Optional[ScheduleConfig] = None
