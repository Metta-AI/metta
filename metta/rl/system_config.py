from typing import ClassVar, Literal

import numpy as np
from omegaconf import DictConfig
from pydantic import ConfigDict, Field

from metta.common.util.collections import remove_none_values
from metta.common.util.config import Config


class SystemConfig(Config):
    vectorization: Literal["serial", "multiprocessing"] = "multiprocessing"
    seed: int = Field(default_factory=lambda: np.random.randint(0, 1000000))
    torch_deterministic: bool = Field(default=True)
    device: str = "cuda"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    @staticmethod
    def MacBookPro() -> "SystemConfig":
        return SystemConfig(
            vectorization="serial",
            device="mps",
        )

def create_system_config(cfg: DictConfig) -> SystemConfig:
    """Create system config from Hydra config.

    Args:
        cfg: The complete Hydra config
    """
    config_dict = remove_none_values(
        {
            "vectorization": cfg.get("vectorization"),
            "seed": cfg.get("seed"),
            "torch_deterministic": cfg.get("torch_deterministic"),
            "device": cfg.get("device"),
        }
    )

    return SystemConfig.model_validate(config_dict)
