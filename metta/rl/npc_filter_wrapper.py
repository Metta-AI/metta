"""
Wrapper for filtering NPC agents from training rewards and observations.

This wrapper ensures that NPC agents don't contribute to training gradients
or metrics, while still being present in the environment for interaction.
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class NPCFilterWrapper:
    """
    Wraps a vectorized environment to filter out NPC agents from training.

    NPCs remain in the environment for interaction but their rewards and
    observations are excluded from training updates.
    """

    def __init__(
        self,
        vecenv: Any,
        num_agents: int,
        policy_agents_pct: float = 1.0,
        npc_group_id: Optional[int] = None,
    ):
        """
        Initialize the NPC filter wrapper.

        Args:
            vecenv: The vectorized environment to wrap
            num_agents: Total number of agents per environment
            policy_agents_pct: Percentage of agents that are policy agents (vs NPCs)
            npc_group_id: Optional group ID for NPCs (for group-based filtering)
        """
        self.vecenv = vecenv
        self.total_agents_per_env = num_agents  # Store original total
        self.policy_agents_pct = policy_agents_pct
        self.npc_group_id = npc_group_id

        # Calculate agent counts
        # Get actual number of environments from vecenv
        self.num_envs = getattr(vecenv, "num_envs", 1)
        if hasattr(vecenv, "envs") and vecenv.envs:
            self.num_envs = len(vecenv.envs)

        self.policy_agents_per_env = max(1, int(num_agents * policy_agents_pct))
        self.npc_agents_per_env = num_agents - self.policy_agents_per_env

        # Create indices for policy vs NPC agents
        self._setup_agent_indices()

        logger.info(
            f"NPCFilterWrapper initialized: {self.num_envs} envs, "
            f"{self.policy_agents_per_env} policy agents, "
            f"{self.npc_agents_per_env} NPC agents per env"
        )

    @property
    def num_agents(self):
        """Return the number of policy agents per environment (filtered count)."""
        return self.policy_agents_per_env

    @property
    def driver_env(self):
        """Proxy to underlying vecenv."""
        return self.vecenv.driver_env

    @property
    def single_action_space(self):
        """Proxy to underlying vecenv."""
        return self.vecenv.single_action_space

    @property
    def single_observation_space(self):
        """Proxy to underlying vecenv."""
        return self.vecenv.single_observation_space

    @property
    def envs(self):
        """Proxy to underlying vecenv."""
        return self.vecenv.envs

    def _setup_agent_indices(self):
        """Setup indices to identify policy vs NPC agents."""
        total_agents = self.total_agents_per_env * self.num_envs

        # For now, use simple index-based assignment
        # Later can be updated to use group IDs after reset
        idx_matrix = torch.arange(total_agents).reshape(self.num_envs, self.total_agents_per_env)

        # Policy agents are the first N agents in each env
        self.policy_idxs = idx_matrix[:, : self.policy_agents_per_env].reshape(-1)
        self.npc_idxs = idx_matrix[:, self.policy_agents_per_env :].reshape(-1)

        # Convert to numpy for indexing
        self.policy_idxs_np = self.policy_idxs.numpy()
        self.npc_idxs_np = self.npc_idxs.numpy()

        # Create mask for filtering
        self.policy_mask = torch.zeros(total_agents, dtype=torch.bool)
        self.policy_mask[self.policy_idxs] = True
        self.policy_mask_np = self.policy_mask.numpy()

    def reset(self, seed: Optional[int] = None):
        """Reset the environment and setup filtering."""
        result = self.vecenv.reset(seed)

        # After reset, we could update indices based on group IDs if available
        # For now, keeping the simple index-based approach

        # Filter observations to only include policy agents
        if isinstance(result, tuple):
            obs, info = result
            filtered_obs = self._filter_observations(obs)
            return filtered_obs, info
        else:
            # Handle different return formats
            return self._filter_observations(result)

    def async_reset(self, seed: Optional[int] = None):
        """Async reset for the environment."""
        return self.vecenv.async_reset(seed)

    def recv(self) -> Tuple[Any, ...]:
        """
        Receive observations and rewards from the environment.
        Filters to only include policy agents.
        """
        o, r, d, t, info, env_id, mask = self.vecenv.recv()

        # Validate indices are within bounds
        actual_size = len(o)
        self._last_actual_size = actual_size  # Save for send()

        if actual_size != len(self.policy_idxs_np) + len(self.npc_idxs_np):
            # Recalculate indices based on actual size
            num_envs_actual = actual_size // self.total_agents_per_env
            idx_matrix = torch.arange(actual_size).reshape(num_envs_actual, self.total_agents_per_env)
            policy_idxs_np = idx_matrix[:, : self.policy_agents_per_env].reshape(-1).numpy()

            # Ensure indices are within bounds
            policy_idxs_np = policy_idxs_np[policy_idxs_np < actual_size]
        else:
            policy_idxs_np = self.policy_idxs_np

        # Filter observations - only policy agents
        o_filtered = o[policy_idxs_np]

        # Filter rewards - only policy agents get rewards for training
        r_filtered = r[policy_idxs_np]

        # Filter dones and truncations
        d_filtered = d[policy_idxs_np]
        t_filtered = t[policy_idxs_np]

        # env_id represents environment indices, not agent indices
        # Keep it unchanged as environments are not filtered, only agents within them
        env_id_filtered = env_id

        # Filter mask
        mask_filtered = mask[policy_idxs_np] if len(mask) > 1 else mask

        # Log filtering stats periodically for debugging
        if hasattr(self, "_recv_count"):
            self._recv_count += 1
        else:
            self._recv_count = 1

        if self._recv_count % 1000 == 0:
            # Log average rewards to verify filtering
            policy_avg = np.mean(r_filtered) if len(r_filtered) > 0 else 0
            npc_avg = np.mean(r[self.npc_idxs_np]) if len(self.npc_idxs_np) > 0 else 0
            logger.debug(
                f"Recv {self._recv_count}: Policy avg reward={policy_avg:.4f}, "
                f"NPC avg reward={npc_avg:.4f} (filtered out)"
            )

        return o_filtered, r_filtered, d_filtered, t_filtered, info, env_id_filtered, mask_filtered

    def send(self, actions: Any):
        """
        Send actions to the environment.
        Need to expand actions to include NPC actions (even though they're random).
        """
        # Actions come in for policy agents only
        # We need to expand to include dummy actions for NPCs

        # Use the actual size from last recv if available
        if hasattr(self, "_last_actual_size"):
            total_agents = self._last_actual_size
        else:
            total_agents = self.total_agents_per_env * self.num_envs

        if isinstance(actions, torch.Tensor):
            # Convert to numpy for pufferlib
            actions = actions.numpy()

        if isinstance(actions, np.ndarray):
            # Create full action array
            full_actions = np.zeros((total_agents, *actions.shape[1:]), dtype=actions.dtype)

            # Recalculate indices if size changed
            if total_agents != len(self.policy_idxs_np) + len(self.npc_idxs_np):
                num_envs_actual = total_agents // self.total_agents_per_env
                idx_matrix = torch.arange(total_agents).reshape(num_envs_actual, self.total_agents_per_env)
                policy_idxs_np = idx_matrix[:, : self.policy_agents_per_env].reshape(-1).numpy()
                policy_idxs_np = policy_idxs_np[policy_idxs_np < total_agents]
            else:
                policy_idxs_np = self.policy_idxs_np

            # Fill in policy agent actions
            full_actions[policy_idxs_np] = actions
            actions_to_send = full_actions
        else:
            # Fallback - just send as is
            actions_to_send = actions

        return self.vecenv.send(actions_to_send)

    def _filter_observations(self, obs: Any) -> Any:
        """Filter observations to only include policy agents."""
        if isinstance(obs, torch.Tensor):
            return obs[self.policy_idxs]
        elif isinstance(obs, np.ndarray):
            return obs[self.policy_idxs_np]
        else:
            # If not tensor/array, return as is
            return obs

    def __getattr__(self, name: str) -> Any:
        """Forward all other attributes to the wrapped environment."""
        return getattr(self.vecenv, name)

    def __len__(self) -> int:
        """Return the number of policy agents (for training)."""
        return len(self.policy_idxs)
