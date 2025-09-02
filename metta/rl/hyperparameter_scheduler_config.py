from metta.common.config import Config


class HyperparameterSchedulerConfig(Config):
    enabled: bool = False
    learning_rate_decay: float = 1.0
    ppo_clip_decay: float = 1.0
