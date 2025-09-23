from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import torch
from tensordict import TensorDict
from torchrl.data import Composite

from metta.common.util.collections import duplicates

if TYPE_CHECKING:
    from metta.rl.loss.loss import Loss
    from metta.rl.trainer_config import TrainerConfig
    from metta.rl.training.training_environment import TrainingEnvironment

logger = logging.getLogger(__name__)


class AlsBuffer:
    """
    A simple, contiguous replay buffer that supports asynchronous rollouts.

    The buffer is structured as a single TensorDict with dimensions
    (num_timesteps, num_parallel_streams), where num_parallel_streams is
    calculated as `num_envs * num_agents * async_factor`. This design
    decouples data storage from sampling logic, allowing for flexible
    sampling strategies to be applied later.
    """

    def __init__(
        self,
        batch_size: int,
        parallel_agents: int,
        num_envs: int,
        experience_spec: Composite,
        device: torch.device | str,
        # Unused, but kept for API compatibility with Experience
        bptt_horizon: int | None = None,
        minibatch_size: int | None = None,
        max_minibatch_size: int | None = None,
        total_agents: int | None = None,
    ):
        """
        Initializes the SimpleReplayBuffer.

        Args:
            batch_size: The total number of timesteps to store in the buffer.
            num_envs: The number of environments.
            num_agents: The number of agents per environment.
            async_factor: The number of asynchronous environment sets.
            experience_spec: A spec defining the data to be stored.
            device: The torch device for tensor allocation.
        """
        self._check_for_duplicate_keys(experience_spec)

        self.batch_size = batch_size
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.num_parallel_streams = parallel_agents

        self.num_timesteps = self.batch_size // self.num_parallel_streams
        if self.batch_size != self.num_timesteps * self.num_parallel_streams:
            logger.info(
                f"batch_size updated from {self.batch_size} to {self.num_timesteps * self.num_parallel_streams}"
            )
            self.batch_size = self.num_timesteps * self.num_parallel_streams

        self.buffer = self._create_buffer(experience_spec)
        self.pos = torch.zeros(self.num_parallel_streams, dtype=torch.long, device=self.device)
        self.is_full = False

        self.num_actors_per_slice = self.num_parallel_streams // num_envs

    def _create_buffer(self, experience_spec: Composite) -> TensorDict:
        """Pre-allocates the TensorDict buffer."""
        buffer_shape = (self.num_timesteps, self.num_parallel_streams)
        return experience_spec.expand(*buffer_shape).to(self.device).zero_()

    def store(self, data_td: TensorDict, env_id: slice):
        """
        Stores a single timestep of experience from a specific environment slice.

        Args:
            data_td: Data for one timestep from `num_actors_per_slice`.
            env_id: The slice of the asynchronous environment.
        """
        # Assuming all streams in an async slice are at the same timestep
        current_pos = self.pos[env_id.start]

        self.buffer[current_pos, env_id] = data_td
        self.pos[env_id] += 1

        if torch.all(self.pos >= self.num_timesteps):
            self.is_full = True

    @staticmethod
    def from_losses(
        trainer_cfg: TrainerConfig,
        env: TrainingEnvironment,
        policy_experience_spec: Composite,
        losses: dict[str, Loss],
        device: torch.device | str,
    ) -> "AlsBuffer":
        """Create experience buffer with merged specs from policy and losses."""
        # Merge all specs
        merged_spec_dict: dict = dict(policy_experience_spec.items())
        for loss in losses.values():
            spec = loss.get_experience_spec()
            merged_spec_dict.update(dict(spec.items()))

        batch_info = env.batch_info

        parallel_agents = getattr(env, "total_parallel_agents", None)
        if parallel_agents is None:
            parallel_agents = batch_info.num_envs * env.meta_data.num_agents

        print(f"parallel_agents: {parallel_agents}")
        print(f"batch_info.num_envs: {batch_info.num_envs}")
        print(f"env.meta_data.num_agents: {env.meta_data.num_agents}")
        print(f"trainer_cfg.batch_size: {trainer_cfg.batch_size}")

        buffer = AlsBuffer(
            batch_size=trainer_cfg.batch_size,
            parallel_agents=parallel_agents,
            num_envs=batch_info.num_envs,
            experience_spec=Composite(merged_spec_dict),
            device=device,
        )
        for loss in losses.values():
            loss.attach_replay_buffer(buffer)
        return buffer

    def sample(self, sample_strategy: Callable, **kwargs: Any) -> Any:
        """
        Samples from the buffer using a provided strategy.

        Args:
            sample_strategy: A function that takes the buffer and returns samples.
            **kwargs: Additional arguments for the sampling strategy.
        """

        return sample_strategy(self.buffer, **kwargs)

    def reset(self):
        """Resets the buffer's position pointers to start filling it again."""
        self.pos.fill_(0)
        self.is_full = False

    def _check_for_duplicate_keys(self, experience_spec: Composite) -> None:
        """Check for duplicate keys in the experience spec."""
        all_keys = list(experience_spec.keys(include_nested=True, leaves_only=True))
        if duplicate_keys := duplicates(all_keys):
            raise ValueError(f"Duplicate keys found in experience_spec: {[str(d) for d in duplicate_keys]}")
