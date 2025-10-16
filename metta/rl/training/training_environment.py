"""Training environment wrapper for vectorized environments."""

import logging
import os
import platform
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Tuple

import numpy as np
import torch
from pydantic import Field
from torch import Tensor

from metta.cogworks.curriculum import Curriculum, CurriculumConfig, env_curriculum
from metta.rl.vecenv import make_vecenv
from metta.utils.batch import calculate_batch_sizes
from mettagrid.base_config import Config
from mettagrid.builder.envs import make_arena
from mettagrid.core import ObsFeature
from mettagrid.mettagrid_c import dtype_actions

logger = logging.getLogger(__name__)


def guess_vectorization() -> Literal["serial", "multiprocessing"]:
    if platform.system() == "Darwin":
        return "serial"
    return "multiprocessing"


class TrainingEnvironmentConfig(Config):
    """Configuration for training environment."""

    curriculum: CurriculumConfig = env_curriculum(make_arena(num_agents=24))
    """Curriculum configuration for task selection"""

    num_workers: int = Field(default=1, ge=1)
    """Number of parallel workers for environment"""

    async_factor: int = Field(default=2, ge=1)
    """Async factor for environment parallelization"""

    auto_workers: bool = Field(default=True)
    """Whether to auto-tune worker count based on available CPU/GPU resources"""

    forward_pass_minibatch_target_size: int = Field(default=4096, gt=0)
    """Target size for forward pass minibatches"""

    zero_copy: bool = Field(default=True)
    """Whether to use zero-copy optimization to avoid memory copies (default assumes multiprocessing)"""

    vectorization: Literal["serial", "multiprocessing"] = Field(default_factory=guess_vectorization)
    """Vectorization mode: 'serial' or 'parallel'"""

    seed: int = Field(default=0)
    """Random seed for environment"""

    write_replays: bool = Field(
        default=False,
        description="Enable writing training episode replays to disk.",
    )
    replay_dir: Path = Field(
        default_factory=lambda: Path("./train_dir/replays/training"),
        description="Base directory where training replays will be stored when writing is enabled.",
    )


@dataclass
class EnvironmentMetaData:
    obs_width: int
    obs_height: int
    obs_features: dict[str, ObsFeature]
    action_names: List[str]
    num_agents: int
    observation_space: Any
    action_space: Any
    feature_normalizations: dict[int, float]


@dataclass
class BatchInfo:
    target_batch_size: int
    batch_size: int
    num_envs: int


class TrainingEnvironment(ABC):
    """Abstract base class for training environment."""

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""

    @abstractmethod
    def get_observations(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[dict], slice, Tensor, int]:
        """Get the observations."""

    @abstractmethod
    def send_actions(self, actions: np.ndarray) -> None:
        """Send the actions."""

    @property
    @abstractmethod
    def batch_info(self) -> BatchInfo:
        """Get the batch information."""

    @property
    @abstractmethod
    def single_action_space(self) -> Any:
        """Get the single action space."""

    @property
    @abstractmethod
    def single_observation_space(self) -> Any:
        """Get the single observation space."""

    @property
    @abstractmethod
    def meta_data(self) -> EnvironmentMetaData:
        """Get the environment metadata."""


class VectorizedTrainingEnvironment(TrainingEnvironment):
    """Manages the vectorized training environment and experience generation."""

    def __init__(self, cfg: TrainingEnvironmentConfig):
        """Initialize training environment."""
        super().__init__()
        self._id = uuid.uuid4().hex[:12]
        self._num_agents = 0
        self._batch_size = 0
        self._num_envs = 0
        self._target_batch_size = 0
        self._num_workers = 0
        self._curriculum = None
        self._vecenv = None

        self._curriculum = Curriculum(cfg.curriculum)
        env_cfg = self._curriculum.get_task().get_env_cfg()
        self._num_agents = env_cfg.game.num_agents

        self._replay_directory: Path | None = None
        if cfg.write_replays:
            base_dir = Path(cfg.replay_dir).expanduser()
            target_dir = (base_dir / self._id).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            self._replay_directory = target_dir

        num_workers = cfg.num_workers
        async_factor = cfg.async_factor

        if cfg.vectorization == "serial":
            num_workers = 1
            async_factor = 1
        else:
            if cfg.auto_workers:
                num_gpus = torch.cuda.device_count() or 1
                cpu_count = os.cpu_count() or 1
                ideal_workers = (cpu_count // 2) // max(num_gpus, 1)
                num_workers = max(1, ideal_workers)

        # Calculate batch sizes
        self._target_batch_size, self._batch_size, self._num_envs = calculate_batch_sizes(
            forward_pass_minibatch_target_size=cfg.forward_pass_minibatch_target_size,
            num_agents=self._num_agents,
            num_workers=num_workers,
            async_factor=async_factor,
        )

        self._num_workers = num_workers

        self._vecenv = make_vecenv(
            self._curriculum,
            cfg.vectorization,
            num_envs=self._num_envs,
            batch_size=self._batch_size,
            num_workers=num_workers,
            zero_copy=cfg.zero_copy,
            is_training=True,
            replay_directory=str(self._replay_directory) if self._replay_directory else None,
        )

        # NOTE: Downstream rollout code currently assumes that PufferLib returns
        # contiguous agent id ranges so we can treat them as a slice; that matches
        # the guarantees when zero_copy=True. If we ever support zero_copy=False,
        # we need to revisit the slice logic in CoreTrainingLoop.rollout_phase.

        # Initialize environment with seed
        self._vecenv.async_reset(cfg.seed)

        self._meta_data = EnvironmentMetaData(
            obs_width=self._vecenv.driver_env.obs_width,
            obs_height=self._vecenv.driver_env.obs_height,
            obs_features=self._vecenv.driver_env.observation_features,
            action_names=self._vecenv.driver_env.action_names,
            num_agents=self._num_agents,
            observation_space=self._vecenv.driver_env.observation_space,
            action_space=self._vecenv.driver_env.single_action_space,
            feature_normalizations=self._vecenv.driver_env.feature_normalizations,
        )

    def __repr__(self) -> str:
        return (
            f"VectorizedTrainingEnvironment("
            f"num_envs={self._num_envs},"
            f"batch_size={self._batch_size},"
            f"target_batch_size={self._target_batch_size},"
            f"num_agents={self._num_agents},"
            f"num_workers={self._num_workers})"
        )

    def close(self) -> None:
        """Close the environment."""
        self._vecenv.close()

    @property
    def meta_data(self) -> EnvironmentMetaData:
        return self._meta_data

    @property
    def batch_info(self) -> BatchInfo:
        return BatchInfo(
            target_batch_size=self._target_batch_size, batch_size=self._batch_size, num_envs=self._num_envs
        )

    @property
    def total_parallel_agents(self) -> int:
        """Total agent slots tracked across all vectorized environments."""
        vecenv_agents = getattr(self._vecenv, "num_agents", None)
        if isinstance(vecenv_agents, int):
            return vecenv_agents
        return self._num_envs * self._num_agents

    @property
    def single_action_space(self) -> Any:
        # Use the underlying driver environment's action space, which remains single-agent Discrete
        return self._vecenv.driver_env.single_action_space

    @property
    def single_observation_space(self) -> Any:
        return self._vecenv.single_observation_space

    @property
    def vecenv(self) -> Any:
        """Return the underlying PufferLib vectorized environment."""
        return self._vecenv

    @property
    def driver_env(self) -> Any:
        """Expose the driver environment for components that need direct access."""
        return self._vecenv.driver_env

    def get_observations(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[dict], slice, Tensor, int]:
        o, r, d, t, info, env_id, mask = self._vecenv.recv()

        training_env_id = slice(env_id[0], env_id[-1] + 1)

        mask = torch.as_tensor(mask)
        num_steps = int(mask.sum().item())

        # Convert to tensors
        o = torch.as_tensor(o)
        r = torch.as_tensor(r)
        d = torch.as_tensor(d)
        t = torch.as_tensor(t)

        return o, r, d, t, info, training_env_id, mask, num_steps

    def send_actions(self, actions: np.ndarray) -> None:
        if actions.dtype != dtype_actions:
            actions = actions.astype(dtype_actions, copy=False)
        self._vecenv.send(actions)
