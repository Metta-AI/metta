"""Dual-policy rollout utilities for training with mixed policy populations."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from metta.agent.policy_store import PolicyStore
from metta.rl.trainer_config import DualPolicyConfig
from metta.rl.util.rollout import run_policy_inference

logger = logging.getLogger(__name__)


class DualPolicyRollout:
    """Handles dual-policy rollout with separate reward tracking."""

    def __init__(
        self,
        config: DualPolicyConfig,
        policy_store: PolicyStore,
        num_agents: int,
        device: torch.device,
    ):
        self.config = config
        self.policy_store = policy_store
        self.num_agents = num_agents
        self.device = device

        # Calculate environment assignments (assuming 2 agents per environment)
        # This is a simplification - in practice you'd need to get the actual num_envs
        self.num_envs = num_agents // 2  # Assuming 2 agents per environment

        # Calculate environment assignments
        self.policy_a_env_count = int(self.num_envs * config.policy_a_percentage)
        self.npc_env_count = self.num_envs - self.policy_a_env_count

        # Create environment assignment masks
        self.policy_a_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.npc_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        if self.policy_a_env_count > 0:
            self.policy_a_mask[: self.policy_a_env_count] = True
        if self.npc_env_count > 0:
            self.npc_mask[self.policy_a_env_count :] = True

        # Initialize NPC policy
        self.npc_policy = None
        self.npc_policy_record = None
        self._initialize_npc_policy()

        # Reward tracking
        self.policy_a_rewards = []
        self.npc_rewards = []
        self.combined_rewards = []

    def _initialize_npc_policy(self):
        """Initialize the NPC policy based on configuration."""
        if not self.config.enabled or self.npc_env_count == 0:
            return

        if self.config.npc_type == "scripted":
            from metta.rl.scripted_npc import create_scripted_npc

            self.npc_policy = create_scripted_npc(self.config.scripted_npc, self.npc_env_count, self.device)
            logger.info(f"Initialized scripted NPC with {self.npc_env_count} agents")

        elif self.config.npc_type == "checkpoint":
            # Support both new checkpoint_npc config and legacy npc_policy_uri
            if self.config.checkpoint_npc:
                # New approach: use checkpoint_path
                checkpoint_path = self.config.checkpoint_npc.checkpoint_path
                if not checkpoint_path:
                    raise ValueError("checkpoint_path must be set in checkpoint_npc config")

                # Load the checkpoint policy
                from metta.rl.util.policy_loader import load_policy_from_checkpoint

                self.npc_policy = load_policy_from_checkpoint(checkpoint_path, self.device)
                logger.info(f"Loaded NPC policy from checkpoint: {checkpoint_path}")

            elif self.config.npc_policy_uri:
                # Legacy approach: use policy URI
                self.npc_policy_record = self.policy_store.policy_record(self.config.npc_policy_uri)
                if not self.npc_policy_record:
                    raise ValueError(f"Could not load NPC policy from {self.config.npc_policy_uri}")

                self.npc_policy = self.npc_policy_record.policy
                logger.info(f"Loaded NPC policy from {self.config.npc_policy_uri}")
            else:
                raise ValueError("Either checkpoint_npc or npc_policy_uri must be set for checkpoint NPCs")

    def run_dual_policy_inference(
        self,
        main_policy: Any,
        observations: Tensor,
        experience: Any,
        training_env_id_start: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        """Run inference with both policies and combine results."""
        if not self.config.enabled:
            # Fall back to single policy inference
            return run_policy_inference(main_policy, observations, experience, training_env_id_start, self.device)

        # Instead of splitting observations, we'll run the main policy on all observations
        # and then replace the actions for NPC environments with scripted actions
        actions, selected_action_log_probs, values, lstm_state = run_policy_inference(
            main_policy, observations, experience, training_env_id_start, self.device
        )

        # For NPC environments, replace actions with scripted actions
        if self.npc_env_count > 0:
            # Get scripted actions for NPC environments
            npc_obs = observations[self.npc_mask]
            npc_actions, _, _, _ = self._get_npc_actions(npc_obs)

            # Replace actions for NPC environments
            actions[self.npc_mask] = npc_actions

            # For scripted NPCs, we don't have proper log probs or values, so we'll use zeros
            # This is a simplification - in practice you might want to handle this differently
            if self.config.npc_type == "scripted":
                # Set log probs and values to zero for NPC environments
                # This is a reasonable approximation since scripted NPCs don't have proper probabilities
                selected_action_log_probs[self.npc_mask] = 0.0
                values[self.npc_mask] = 0.0

        return actions, selected_action_log_probs, values, lstm_state

    def _get_npc_actions(self, observations: Tensor) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        """Get actions from the NPC policy."""
        if self.config.npc_type == "scripted":
            # Use scripted NPC behavior
            if self.npc_policy is None:
                raise ValueError("Scripted NPC policy not initialized")

            actions = self.npc_policy.get_actions(observations)
            # For scripted NPCs, return dummy values for log_probs, values, and lstm_state
            log_probs = torch.zeros(observations.shape[0], device=self.device)
            values = torch.zeros(observations.shape[0], device=self.device)
            lstm_state = None

        elif self.config.npc_type == "checkpoint":
            # Use the same inference function as the main policy
            if self.npc_policy is None:
                raise ValueError("NPC policy not initialized")

            # For checkpoint policies, we need to create a proper experience object
            # We'll use the same structure as the main policy but with the NPC observations
            import numpy as np

            from metta.rl.experience import Experience

            # Get the observation and action spaces from the main policy's experience
            # We'll create a minimal experience object for the NPC batch
            npc_batch_size = observations.shape[0]

            # Create a minimal experience object for the NPC batch
            # We need to infer the spaces from the observations
            obs_shape = observations.shape[1:]  # Remove batch dimension
            atn_shape = (2,)  # Action shape is (action_type, action_param)

            # Create a simple space-like object for the observations
            class SimpleSpace:
                def __init__(self, shape, dtype):
                    self.shape = shape
                    # Convert PyTorch dtype to NumPy dtype for compatibility
                    if hasattr(dtype, "numpy"):
                        self.dtype = dtype.numpy()
                    elif dtype == torch.int32:
                        self.dtype = np.int32
                    elif dtype == torch.int64:
                        self.dtype = np.int64
                    elif dtype == torch.float32:
                        self.dtype = np.float32
                    elif dtype == torch.float64:
                        self.dtype = np.float64
                    else:
                        self.dtype = np.float32  # Default fallback

            obs_space = SimpleSpace(obs_shape, observations.dtype)
            atn_space = SimpleSpace(atn_shape, np.int32)  # Use NumPy dtype directly

            # Get LSTM configuration from the NPC policy
            from metta.rl.util.rollout import get_lstm_config

            hidden_size, num_lstm_layers = get_lstm_config(self.npc_policy)

            npc_experience = Experience(
                total_agents=npc_batch_size,
                batch_size=npc_batch_size,
                bptt_horizon=1,
                minibatch_size=npc_batch_size,
                max_minibatch_size=npc_batch_size,
                obs_space=obs_space,
                atn_space=atn_space,
                device=self.device,
                hidden_size=hidden_size,
                num_lstm_layers=num_lstm_layers,
            )

            # Use the same inference function as the main policy
            actions, log_probs, values, lstm_state = run_policy_inference(
                self.npc_policy, observations, npc_experience, 0, self.device
            )

        else:
            raise ValueError(f"Unknown NPC type: {self.config.npc_type}")

        return actions, log_probs, values, lstm_state

    def track_rewards(self, rewards: Tensor):
        """Track rewards separately for each policy type."""
        if not self.config.enabled:
            return

        # Split rewards by policy
        policy_a_reward = rewards[self.policy_a_mask].sum().item() if self.policy_a_env_count > 0 else 0.0
        npc_reward = rewards[self.npc_mask].sum().item() if self.npc_env_count > 0 else 0.0
        combined_reward = rewards.sum().item()

        # Store for later aggregation
        self.policy_a_rewards.append(policy_a_reward)
        self.npc_rewards.append(npc_reward)
        self.combined_rewards.append(combined_reward)

        # Debug logging every 100 steps
        if len(self.policy_a_rewards) % 100 == 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Dual-policy rewards tracked - Policy A: {policy_a_reward:.3f}, "
                f"NPC: {npc_reward:.3f}, Combined: {combined_reward:.3f}"
            )

    def get_reward_stats(self) -> Dict[str, float]:
        """Get aggregated reward statistics."""
        if not self.config.enabled or not self.policy_a_rewards:
            return {}

        # Calculate averages
        avg_policy_a_reward = sum(self.policy_a_rewards) / len(self.policy_a_rewards)
        avg_npc_reward = sum(self.npc_rewards) / len(self.npc_rewards)
        avg_combined_reward = sum(self.combined_rewards) / len(self.combined_rewards)

        # Calculate totals
        total_policy_a_reward = sum(self.policy_a_rewards)
        total_npc_reward = sum(self.npc_rewards)
        total_combined_reward = sum(self.combined_rewards)

        return {
            "policy_a_reward": avg_policy_a_reward,
            "policy_a_reward_total": total_policy_a_reward,
            "npc_reward": avg_npc_reward,
            "npc_reward_total": total_npc_reward,
            "combined_reward": avg_combined_reward,
            "combined_reward_total": total_combined_reward,
            "policy_a_agent_count": self.policy_a_env_count,
            "npc_agent_count": self.npc_env_count,
        }

    def reset_reward_tracking(self):
        """Reset reward tracking for new episode."""
        self.policy_a_rewards.clear()
        self.npc_rewards.clear()
        self.combined_rewards.clear()

    def get_agent_assignments(self) -> Dict[str, Any]:
        """Get information about agent assignments."""
        return {
            "policy_a_count": self.policy_a_env_count,
            "npc_count": self.npc_env_count,
            "policy_a_percentage": self.config.policy_a_percentage,
            "npc_type": self.config.npc_type,
            "enabled": self.config.enabled,
        }
