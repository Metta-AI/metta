"""
MettaGridBenchMARLEnv - BenchMARL adapter for MettaGrid.

This class implements the BenchMARL/TorchRL interface using the base MettaGridCore,
following the same pattern as other environment adapters (Gym, PettingZoo).
Also includes the MettaGridTask class required by BenchMARL.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from benchmarl.environments import TaskClass
from tensordict import TensorDict
from torchrl.data import CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from typing_extensions import override

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import dtype_actions


class MettaGridBenchMARLEnv(MettaGridCore, EnvBase):
    """
    BenchMARL/TorchRL adapter for MettaGrid environments.

    This class provides a TorchRL-compatible interface for MettaGrid environments,
    enabling integration with the BenchMARL benchmarking framework for multi-agent
    reinforcement learning. No training features are included - this is purely
    for BenchMARL/TorchRL compatibility.

    Inherits from:
    - MettaGridCore: Core C++ environment wrapper functionality
    - torchrl.envs.EnvBase: TorchRL environment interface required by BenchMARL
    """

    def __init__(
        self,
        mg_config: MettaGridConfig,
        device: Union[str, torch.device] = "cpu",
        batch_size: Optional[torch.Size] = None,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize BenchMARL/TorchRL environment.

        Args:
            mg_config: Environment configuration
            device: Device to run the environment on
            batch_size: Batch size for vectorized environments
            render_mode: Rendering mode
            **kwargs: Additional arguments
        """
        # Initialize core environment (no training features)
        MettaGridCore.__init__(
            self,
            mg_config,
            render_mode=render_mode,
        )

        # Set device
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device

        # Initialize TorchRL base
        EnvBase.__init__(
            self,
            device=device,
            batch_size=batch_size or torch.Size([]),
        )

        # Cache agent information
        self._agent_names: List[str] = []
        self._setup_agents()

        # Set up specs
        self._setup_specs()

    def _setup_agents(self) -> None:
        """Setup agent names based on number of agents."""
        num_agents = self.num_agents
        self._agent_names = [f"agent_{i}" for i in range(num_agents)]

    def _setup_specs(self) -> None:
        """Set up TensorSpec specifications for observations, actions, and rewards."""
        # Get spaces from core environment
        obs_space = self._observation_space
        action_space = self._action_space

        # Create observation spec - group-level only for simplicity
        obs_spec_dict = {
            "agents": UnboundedContinuousTensorSpec(
                shape=(self.num_agents, *obs_space.shape),
                dtype=torch.float32,
                device=self._device,
            )
        }

        # Create action spec - flatten multi-discrete to single discrete for simplicity
        # Each agent has a multi-discrete action space, but we can represent it as vectors
        action_spec_dict = {
            "agents": DiscreteTensorSpec(
                n=int(action_space.nvec.max()),  # Use max to be safe
                shape=(self.num_agents, len(action_space.nvec)),
                dtype=torch.long,
                device=self._device,
            )
        }

        # Create reward spec (use different keys to avoid collisions)
        reward_spec_dict = {
            "reward": UnboundedContinuousTensorSpec(
                shape=(self.num_agents,),
                dtype=torch.float32,
                device=self._device,
            )
        }

        # Create done spec (use different keys to avoid collisions)
        done_spec_dict = {
            "done": DiscreteTensorSpec(
                n=2,
                shape=(self.num_agents,),
                dtype=torch.bool,
                device=self._device,
            ),
            "terminated": DiscreteTensorSpec(
                n=2,
                shape=(self.num_agents,),
                dtype=torch.bool,
                device=self._device,
            ),
        }

        # Set specs
        self.observation_spec = CompositeSpec(obs_spec_dict)
        self.action_spec = CompositeSpec(action_spec_dict)
        self.reward_spec = CompositeSpec(reward_spec_dict)
        self.done_spec = CompositeSpec(done_spec_dict)

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> TensorDict:
        """
        Reset the environment and return initial observations.

        Args:
            seed: Random seed for reset
            **kwargs: Additional reset arguments

        Returns:
            TensorDict with initial observations
        """
        # Reset core environment using MettaGridCore method
        obs_array, info = MettaGridCore.reset(self, seed)

        # Convert to TensorDict format
        data = {
            "agents": torch.tensor(obs_array, dtype=torch.float32, device=self._device),
            "done": torch.zeros(self.num_agents, dtype=torch.bool, device=self._device),
            "terminated": torch.zeros(self.num_agents, dtype=torch.bool, device=self._device),
        }

        return TensorDict(data, batch_size=self.batch_size, device=self._device)

    @override  # torchrl.envs.EnvBase._reset
    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs: Any) -> TensorDict:
        """
        Reset the environment and return initial observations.

        Args:
            tensordict: Optional tensordict with reset parameters
            **kwargs: Additional reset arguments

        Returns:
            TensorDict with initial observations
        """
        # Extract seed if provided
        seed = kwargs.get("seed", None)
        if tensordict is not None and "seed" in tensordict.keys():
            seed = tensordict["seed"].item()

        # Reset core environment using MettaGridCore method
        obs_array, info = MettaGridCore.reset(self, seed)

        # Convert to TensorDict format
        data = {
            "agents": torch.tensor(obs_array, dtype=torch.float32, device=self._device),
            "done": torch.zeros(self.num_agents, dtype=torch.bool, device=self._device),
            "terminated": torch.zeros(self.num_agents, dtype=torch.bool, device=self._device),
        }

        return TensorDict(data, batch_size=self.batch_size, device=self._device)

    def step(self, action: TensorDict) -> TensorDict:
        """
        Execute one timestep of the environment dynamics.

        Args:
            action: TensorDict with actions

        Returns:
            TensorDict with observations, rewards, and done flags
        """
        return self._step(action)

    @override  # torchrl.envs.EnvBase._step
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Execute one timestep of the environment dynamics.

        Args:
            tensordict: TensorDict with actions

        Returns:
            TensorDict with observations, rewards, and done flags
        """
        # Extract actions from tensordict
        if "agents" in tensordict.keys():
            actions = tensordict["agents"].cpu().numpy().astype(dtype_actions)
        else:
            # Fallback to individual agent actions
            actions = np.zeros((self.num_agents, 2), dtype=dtype_actions)
            for i, agent_name in enumerate(self._agent_names):
                if agent_name in tensordict.keys():
                    actions[i] = tensordict[agent_name].cpu().numpy()

        # Step environment
        obs, rewards, terminals, truncations, infos = super().step(actions)

        # Convert to TensorDict format
        data = {
            "agents": torch.tensor(obs, dtype=torch.float32, device=self._device),
            "reward": torch.tensor(rewards, dtype=torch.float32, device=self._device),
            "done": torch.tensor(terminals | truncations, dtype=torch.bool, device=self._device),
            "terminated": torch.tensor(terminals, dtype=torch.bool, device=self._device),
        }

        return TensorDict(data, batch_size=self.batch_size, device=self._device)

    @override  # torchrl.envs.EnvBase._set_seed
    def _set_seed(self, seed: Optional[int]) -> int:
        """
        Set environment seed.

        Args:
            seed: Random seed

        Returns:
            The seed that was set
        """
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        self._current_seed = seed
        return seed

    @override  # torchrl.envs.EnvBase.close
    def close(self) -> None:
        """Close the environment."""
        super().close()

    # BenchMARL compatibility properties
    @property
    def group_map(self) -> Dict[str, List[str]]:
        """Get agent group mapping for BenchMARL."""
        return {"agents": self._agent_names}

    @property
    def agent_names(self) -> List[str]:
        """Get list of agent names."""
        return self._agent_names

    @property
    def max_num_agents(self) -> int:
        """Get maximum number of agents."""
        return self.num_agents


class MettaGridTask(TaskClass):
    """
    BenchMARL task implementation for MettaGrid environments.

    This class provides the interface required by BenchMARL to run
    standardized MARL benchmarks on MettaGrid environments.
    """

    def __init__(
        self,
        mg_config: MettaGridConfig,
        task_name: str = "mettagrid_default",
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize BenchMARL task.

        Args:
            mg_config: MettaGrid configuration
            task_name: Name of the task
            max_steps: Maximum steps per episode (defaults to config value)
            render_mode: Rendering mode
            **kwargs: Additional arguments
        """
        # Store MettaGrid-specific config
        self._mg_config = mg_config
        self._max_steps = max_steps or mg_config.game.max_steps
        self._render_mode = render_mode
        self._kwargs = kwargs

        # Create task config dict for BenchMARL
        task_config = {
            "mg_config": mg_config,
            "max_steps": self._max_steps,
            "render_mode": render_mode,
            **kwargs,
        }

        # Initialize TaskClass with required name and config
        super().__init__(name=task_name, config=task_config)

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: Union[str, torch.device],
    ) -> Callable[[], EnvBase]:
        """
        Get environment factory function.

        Args:
            num_envs: Number of parallel environments
            continuous_actions: Whether to use continuous actions
            seed: Random seed
            device: Device to run on

        Returns:
            Factory function that creates TorchRL environment
        """
        if continuous_actions:
            raise ValueError("MettaGrid only supports discrete actions")

        def make_env() -> EnvBase:
            env = MettaGridBenchMARLEnv(
                mg_config=self._mg_config,
                device=device,
                batch_size=torch.Size([num_envs]) if num_envs > 1 else torch.Size([]),
                render_mode=self._render_mode,
                **self._kwargs,
            )
            if seed is not None:
                env.set_seed(seed)
            return env

        return make_env

    def supports_continuous_actions(self) -> bool:
        """Check if task supports continuous actions."""
        return False

    def supports_discrete_actions(self) -> bool:
        """Check if task supports discrete actions."""
        return True

    def max_steps(self, env: EnvBase) -> int:
        """Get maximum steps for evaluation."""
        return self._max_steps

    def has_render(self, env: EnvBase) -> bool:
        """Check if environment supports rendering."""
        return self._render_mode is not None

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        """Get agent group mapping."""
        if hasattr(env, "group_map"):
            return env.group_map
        # Fallback for environments without explicit group mapping
        return {"agents": env.agent_names if hasattr(env, "agent_names") else []}

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        """Get observation specification."""
        return env.observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        """Get action specification."""
        return env.action_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        """Get action mask specification (not implemented for MettaGrid)."""
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        """Get info specification (optional)."""
        return None

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        """Get global state specification (optional)."""
        # Could potentially return a spec for the full grid state
        return None

    @staticmethod
    def env_name() -> str:
        """Get environment name."""
        return "mettagrid"
