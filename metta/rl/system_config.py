import os
import platform
from typing import ClassVar, Literal

import numpy as np
import torch
from pydantic import ConfigDict, Field

from metta.common.util.config import Config


def guess_device() -> str:
    if platform.system() == "Darwin":
        return "mps"
    if not torch.cuda.is_available():
        return "cpu"

    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

    return f"cuda:{local_rank}"


def guess_vectorization() -> Literal["serial", "multiprocessing"]:
    if platform.system() == "Darwin":
        return "serial"
    return "multiprocessing"


class SystemConfig(Config):
    vectorization: Literal["serial", "multiprocessing"] = Field(default_factory=guess_vectorization)
    seed: int = Field(default_factory=lambda: np.random.randint(0, 1000000))
    torch_deterministic: bool = Field(default=True)
    device: str = Field(default_factory=guess_device)
    data_dir: str = Field(default="./train_dir")

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )
