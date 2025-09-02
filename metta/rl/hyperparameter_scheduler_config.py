from metta.common.config import Config


class HyperparameterSchedulerConfig(Config):
    enabled: bool = False
    learning_rate_decay: float = 1.0  # 1.0 = no decay, 0.95 = decay to 95% each epoch
    ppo_clip_decay: float = 1.0  # 1.0 = no decay, 0.9 = decay to 90% each epoch
