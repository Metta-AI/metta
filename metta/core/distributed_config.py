"""Distributed configuration for Metta."""

import logging
from typing import ClassVar

import torch
from pydantic import ConfigDict, Field

from metta.common.util.config import Config

logger = logging.getLogger(__name__)


class DistributedConfig(Config):
    world_size: int = Field(default_factory=lambda: torch.distributed.get_world_size())
    rank: int = Field(default_factory=lambda: torch.distributed.get_rank())
    is_master: bool = Field(default_factory=lambda: torch.distributed.get_rank() == 0)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )
