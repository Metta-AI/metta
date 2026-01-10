import os
import platform
from datetime import timedelta
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from pydantic import ConfigDict, Field

from metta.common.cuda_utils import is_cuda_supported
from mettagrid.base_config import Config


def guess_device() -> str:
    if platform.system() == "Darwin":
        return "mps"

    if not is_cuda_supported():
        return "cpu"

    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

    return f"cuda:{local_rank}"


def guess_vectorization() -> Literal["serial", "multiprocessing"]:
    if platform.system() == "Darwin":
        return "serial"
    return "multiprocessing"


def guess_data_dir() -> Path:
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    return Path("./train_dir")


class SystemConfig(Config):
    vectorization: Literal["serial", "multiprocessing"] = Field(default_factory=guess_vectorization)
    seed: int = Field(default_factory=lambda: np.random.randint(0, 1000000))
    torch_deterministic: bool = Field(default=True)
    device: str = Field(default_factory=guess_device)
    data_dir: Path = Field(default_factory=guess_data_dir)
    local_only: bool = Field(default=False)
    nccl_timeout: timedelta = Field(default=timedelta(minutes=10))

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )
