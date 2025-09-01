from typing import Optional, Union

from pydantic import Field

from metta.common.config import Config


class ConstantScheduleConfig(Config):
    """Configuration for constant schedule."""

    type: str = "constant"
    initial_value: float = Field(description="Constant value to maintain")


class LinearScheduleConfig(Config):
    """Configuration for linear schedule."""

    type: str = "linear"
    initial_value: float = Field(description="Starting value")
    min_value: Optional[float] = Field(default=None, description="Minimum value (defaults to 10% of initial)")


class CosineScheduleConfig(Config):
    """Configuration for cosine schedule."""

    type: str = "cosine"
    initial_value: float = Field(description="Starting value")
    min_value: Optional[float] = Field(default=None, description="Minimum value (defaults to 10% of initial)")


class ExponentialScheduleConfig(Config):
    """Configuration for exponential schedule."""

    type: str = "exponential"
    initial_value: float = Field(description="Starting value")
    decay_rate: float = Field(default=0.95, description="Decay rate per progress unit")
    min_value: Optional[float] = Field(default=None, description="Minimum value (defaults to 10% of initial)")


class LogarithmicScheduleConfig(Config):
    """Configuration for logarithmic schedule."""

    type: str = "logarithmic"
    initial_value: float = Field(description="Starting value")
    min_value: Optional[float] = Field(default=None, description="Minimum value (defaults to 10% of initial)")
    decay_rate: float = Field(default=0.1, description="Decay rate for logarithmic scaling")


# Union type for all schedule configurations
ScheduleConfig = Union[
    ConstantScheduleConfig,
    LinearScheduleConfig,
    CosineScheduleConfig,
    ExponentialScheduleConfig,
    LogarithmicScheduleConfig,
]


class HyperparameterSchedulerConfig(Config):
    """
    Configuration for hyperparameter scheduling in RL training.
    """

    learning_rate_schedule: Optional[ScheduleConfig] = Field(
        default=ConstantScheduleConfig(type="constant", initial_value=0.000457), description="Learning rate scheduling configuration"
    )
    ppo_clip_schedule: Optional[ScheduleConfig] = Field(
        default=ConstantScheduleConfig(type="constant", initial_value=0.3), description="PPO clip coefficient scheduling configuration"
    )
    ppo_ent_coef_schedule: Optional[ScheduleConfig] = Field(
        default=ConstantScheduleConfig(type="constant", initial_value=0.0021), description="PPO entropy coefficient scheduling configuration"
    )
    ppo_vf_clip_schedule: Optional[ScheduleConfig] = Field(
        default=ConstantScheduleConfig(type="constant", initial_value=0.3), description="PPO value function clip coefficient scheduling configuration"
    )
    ppo_l2_reg_loss_schedule: Optional[ScheduleConfig] = Field(
        default=ConstantScheduleConfig(type="constant", initial_value=0), description="PPO L2 regularization loss coefficient scheduling configuration"
    )
    ppo_l2_init_loss_schedule: Optional[ScheduleConfig] = Field(
        default=ConstantScheduleConfig(type="constant", initial_value=0), description="PPO L2 initialization loss coefficient scheduling configuration"
    )
