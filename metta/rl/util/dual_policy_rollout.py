"""Dual-policy rollout utilities for training with mixed agent populations."""

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
            if not self.config.npc_policy_uri:
                raise ValueError("npc_policy_uri must be set for checkpoint NPCs")

            self.npc_policy_record = self.policy_store.policy_record(self.config.npc_policy_uri)
            if not self.npc_policy_record:
                raise ValueError(f"Could not load NPC policy from {self.config.npc_policy_uri}")

            self.npc_policy = self.npc_policy_record.policy
            logger.info(f"Loaded NPC policy from {self.config.npc_policy_uri}")

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
            # Scripted NPCs don't return log probs or values, so we create dummy ones
            if self.npc_policy is None:
                raise ValueError("NPC policy not initialized")
            actions = self.npc_policy.get_actions(observations)
            log_probs = torch.zeros(int(observations.shape[0]), device=self.device)
            values = torch.zeros(int(observations.shape[0]), device=self.device)
            lstm_state = None

        elif self.config.npc_type == "checkpoint":
            # Use the same inference function as the main policy
            if self.npc_policy is None:
                raise ValueError("NPC policy not initialized")
            # Create a dummy experience object for checkpoint policies
            from metta.rl.experience import Experience

            dummy_experience = Experience(
                total_agents=observations.shape[0],
                batch_size=observations.shape[0],
                bptt_horizon=1,
                minibatch_size=observations.shape[0],
                max_minibatch_size=observations.shape[0],
                obs_space=None,  # Will be set by the policy
                atn_space=None,  # Will be set by the policy
                device=self.device,
                hidden_size=256,  # Default value
            )
            # Type check for checkpoint policies
            if hasattr(self.npc_policy, "forward"):
                actions, log_probs, values, lstm_state = run_policy_inference(
                    self.npc_policy, observations, dummy_experience, 0, self.device
                )
            else:
                raise ValueError("Checkpoint NPC policy must be a PyTorch module")

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
