from metta.mettagrid.config import Config
from metta.rl.trainer_config import TrainerConfig


class SweepConfig(Config):
    """Configuration for a sweep."""

    trainer_cfg: TrainerConfig
