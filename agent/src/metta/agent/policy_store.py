"""This file implements a PolicyStore class that manages loading and caching of trained policies.
It provides functionality to:
- Load policies from local files or remote URIs
- Cache loaded policies to avoid reloading
- Select policies based on metadata filters
- Track policy metadata and versioning

The PolicyStore is used by the training system to manage opponent policies and checkpoints."""

import logging
from typing import Literal

from omegaconf import DictConfig

from metta.common.config import Config
from metta.common.wandb.wandb_context import WandbRun

logger = logging.getLogger("policy_store")


PolicySelectorType = Literal["all", "top", "latest", "rand"]


class PolicySelectorConfig(Config):
    type: PolicySelectorType = "top"
    metric: str = "score"


class PolicyMissingError(ValueError):
    pass


class PolicyStore:
    def __init__(
        self,
        device: str | None = None,  # for loading policies from checkpoints
        wandb_run: WandbRun | None = None,  # for saving artifacts to wandb
        data_dir: str | None = None,  # for storing policy artifacts locally for cached access
        wandb_entity: str | None = None,  # for loading policies from wandb
        wandb_project: str | None = None,  # for loading policies from wandb
        pytorch_cfg: DictConfig | None = None,  # for loading pytorch policies
        policy_cache_size: int = 10,  # num policies to keep in memory
    ) -> None:
        pass
