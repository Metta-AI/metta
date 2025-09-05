from metta.mettagrid.config import Config


class HyperparameterSchedulerConfig(Config):
    enabled: bool = False
    schedule_type: str = "exponential"
    learning_rate_decay: float = 1.0
    ppo_clip_decay: float = 1.0
    ppo_ent_coef_decay: float = 1.0
