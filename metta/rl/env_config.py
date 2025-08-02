from typing import ClassVar, Literal

import numpy as np
from omegaconf import DictConfig
from pydantic import ConfigDict, Field

from metta.common.util.collections import remove_none_values
from metta.common.util.typed_config import BaseModelWithForbidExtra

DeviceType = Literal["cpu", "cuda"]


class EnvConfig(BaseModelWithForbidExtra):
    vectorization: Literal["serial", "multiprocessing"] = "multiprocessing"
    seed: int = Field(default_factory=lambda: np.random.randint(0, 1000000))
    torch_deterministic: bool = Field(default=True)
    device: DeviceType = "cuda"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


def create_env_config(cfg: DictConfig) -> EnvConfig:
    """Create env config from Hydra config.

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

    return EnvConfig.model_validate(config_dict)
