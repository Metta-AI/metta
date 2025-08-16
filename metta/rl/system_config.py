from typing import ClassVar, Literal

import numpy as np
from pydantic import ConfigDict, Field

from metta.common.util.config import Config


class SystemConfig(Config):
    vectorization: Literal["serial", "multiprocessing"] = "multiprocessing"
    seed: int = Field(default_factory=lambda: np.random.randint(0, 1000000))
    torch_deterministic: bool = Field(default=True)
    device: str = "cuda"
    data_dir: str = Field(default="./train_dir")

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
