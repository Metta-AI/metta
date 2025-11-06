import datetime
import os
import pathlib
import platform
import random
import typing

import numpy as np
import pydantic
import torch

import mettagrid.base_config


def guess_device() -> str:
    if platform.system() == "Darwin":
        return "mps"
    if not torch.cuda.is_available():
        return "cpu"

    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

    return f"cuda:{local_rank}"


def guess_vectorization() -> typing.Literal["serial", "multiprocessing"]:
    if platform.system() == "Darwin":
        return "serial"
    return "multiprocessing"


def guess_data_dir() -> pathlib.Path:
    if os.environ.get("DATA_DIR"):
        return pathlib.Path(os.environ["DATA_DIR"])
    return pathlib.Path("./train_dir")


class SystemConfig(mettagrid.base_config.Config):
    vectorization: typing.Literal["serial", "multiprocessing"] = pydantic.Field(default_factory=guess_vectorization)
    seed: int = pydantic.Field(default_factory=lambda: np.random.randint(0, 1000000))
    torch_deterministic: bool = pydantic.Field(default=True)
    device: str = pydantic.Field(default_factory=guess_device)
    data_dir: pathlib.Path = pydantic.Field(default_factory=guess_data_dir)
    remote_prefix: str | None = pydantic.Field(default=None)
    local_only: bool = pydantic.Field(default=False)
    nccl_timeout: datetime.timedelta = pydantic.Field(default=datetime.timedelta(minutes=10))

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


def seed_everything(system_cfg: SystemConfig):
    # Despite these efforts, we still don't get deterministic behavior. But presumably
    # this is better than nothing.
    # https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility
    rank = int(os.environ.get("RANK", 0))

    seed = system_cfg.seed
    torch_deterministic = system_cfg.torch_deterministic

    # Add rank offset to base seed for distributed training to ensure different
    # processes generate uncorrelated random sequences
    rank_specific_seed = (seed + rank) if seed is not None else rank

    random.seed(rank_specific_seed)
    np.random.seed(rank_specific_seed)
    if seed is not None:
        torch.manual_seed(rank_specific_seed)
        torch.cuda.manual_seed_all(rank_specific_seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic
    torch.use_deterministic_algorithms(torch_deterministic)

    if torch_deterministic:
        # Set CuBLAS workspace config for deterministic behavior on CUDA >= 10.2
        # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
