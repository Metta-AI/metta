"""Dual policy NPC training support.

This module encapsulates all dual policy functionality to enable training
with both a learning policy and a fixed NPC policy simultaneously.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from torch import Tensor

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore

if TYPE_CHECKING:
    from metta.mettagrid import MettaGridEnv

logger = logging.getLogger(__name__)


class DualPolicyHandler:
    """Handles all dual policy NPC logic for training with mixed policies."""

    def __init__(
        self,
        enabled: bool = False,
        training_agents_pct: float = 0.5,
        checkpoint_npc: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize dual policy handler.

        Args:
            enabled: Whether dual policy training is enabled
            training_agents_pct: Percentage of agents using training policy (0.0 to 1.0)
            checkpoint_npc: URI/path to NPC policy checkpoint
            device: Device for tensor operations
        """
        self.enabled = enabled
        self.training_agents_pct = training_agents_pct
        self.checkpoint_npc = checkpoint_npc
        self.device = device if device is not None else torch.device("cpu")
        self.npc_policy: Optional[PolicyAgent] = None
        self.npc_record: Optional[PolicyRecord] = None
        self._agent_groups: Optional[Dict[int, int]] = None
        # Legacy mask fields retained for compatibility but not used in slicing mode
        self._training_mask: Optional[Tensor] = None
        self._npc_mask: Optional[Tensor] = None
        # Cached split counts
        self._num_agents: int | None = None
        self._num_training: int | None = None
        self._num_npc: int | None = None

    def load_npc_policy(
        self,
        policy_store: PolicyStore,
        metta_grid_env: MettaGridEnv,
    ) -> None:
        """Load NPC policy from checkpoint.

        Args:
            policy_store: Policy store for loading checkpoints
            metta_grid_env: Environment for policy initialization
        """
        if not self.enabled or not self.checkpoint_npc:
            return

        try:
            logger.info(f"Loading NPC policy from {self.checkpoint_npc}")

            # Load the NPC policy record
            self.npc_record = policy_store.load_from_uri(self.checkpoint_npc)
            if not self.npc_record:
                logger.warning(f"Failed to load NPC policy from {self.checkpoint_npc}")
                self.enabled = False
                return

            self.npc_policy = self.npc_record.policy
            self.npc_policy.to(self.device)
            self.npc_policy.eval()  # Set to evaluation mode

            # Initialize NPC policy for environment
            from metta.rl.policy_management import initialize_policy_for_environment

            initialize_policy_for_environment(
                policy_record=self.npc_record,
                metta_grid_env=metta_grid_env,
                device=self.device,
                restore_feature_mapping=True,
            )

            logger.info(f"Successfully loaded NPC policy: {self.npc_record.run_name}")

        except Exception as e:
            logger.error(f"Failed to load NPC policy: {e}", exc_info=True)
            self.enabled = False

    def setup_agent_groups(self, env: MettaGridEnv, ensure_each: bool = False) -> None:
        """Setup training vs NPC agent groups.

        Args:
            env: MettaGrid environment
            ensure_each: If True, ensure at least one agent in each group
        """
        if not self.enabled:
            return

        num_agents = env.num_agents
        # Deterministic contiguous split: first NPC, rest training
        num_training = max(1, int(num_agents * self.training_agents_pct))
        if ensure_each:
            num_training = max(1, min(num_agents - 1, num_training))
        num_npc = num_agents - num_training

        self._num_agents = num_agents
        self._num_training = num_training
        self._num_npc = num_npc

        # 0 = training, 1 = NPC; indices [0:num_npc) are NPC
        self._agent_groups = {i: (1 if i < num_npc else 0) for i in range(num_agents)}

        logger.debug(f"Agent groups (contiguous): {num_training} training, {num_npc} NPC")

    def initialize_agent_split(self, num_agents: int, ensure_each: bool = False) -> None:
        """Initialize contiguous agent split without relying on env instance.

        First agents [0:num_npc) are NPCs, remaining are training.
        """
        if not self.enabled:
            return
        num_training = max(1, int(num_agents * self.training_agents_pct))
        if ensure_each:
            num_training = max(1, min(num_agents - 1, num_training))
        num_npc = max(0, num_agents - num_training)

        self._num_agents = num_agents
        self._num_training = num_training
        self._num_npc = num_npc
        self._agent_groups = {i: (1 if i < num_npc else 0) for i in range(num_agents)}

    def create_agent_masks(self, num_agents: int, num_envs: int) -> Tuple[Tensor, Tensor]:
        """Deprecated: masks are not used in slicing mode. Kept for compatibility."""
        training_mask = torch.ones(num_envs, num_agents, dtype=torch.bool, device=self.device)
        npc_mask = torch.zeros(num_envs, num_agents, dtype=torch.bool, device=self.device)
        self._training_mask = training_mask
        self._npc_mask = npc_mask
        return training_mask, npc_mask

    def compute_step_stats(self, env: MettaGridEnv, steps: int) -> Dict[str, Any]:
        """Compute per-step dual policy statistics.

        Args:
            env: MettaGrid environment
            steps: Current step count

        Returns:
            Dictionary of dual policy statistics
        """
        if not self.enabled or not self._agent_groups:
            return {}

        stats = {}

        # Get grid objects and compute stats per group
        grid_objects = env.grid_objects

        # Count agents and compute aggregate stats per group
        group_stats = {0: {}, 1: {}}  # 0 = training, 1 = NPC
        group_counts = {0: 0, 1: 0}

        for _, obj_data in grid_objects.items():
            if obj_data.get("type") == 0:  # Agent type
                agent_id = obj_data.get("agent_id", -1)
                if agent_id in self._agent_groups:
                    group = self._agent_groups[agent_id]
                    group_counts[group] += 1

                    # Aggregate health/hearts
                    hearts = obj_data.get("agent:heart", 0)
                    if "total_hearts" not in group_stats[group]:
                        group_stats[group]["total_hearts"] = 0
                    group_stats[group]["total_hearts"] += hearts

        # Add to stats
        stats["dual_policy"] = {
            "training_agents": group_counts[0],
            "npc_agents": group_counts[1],
            "training_avg_hearts": group_stats[0].get("total_hearts", 0) / max(1, group_counts[0]),
            "npc_avg_hearts": group_stats[1].get("total_hearts", 0) / max(1, group_counts[1]),
            "step": steps,
        }

        return stats

    def compute_episode_stats(self, env: MettaGridEnv, episode_rewards: np.ndarray, infos: Dict[str, Any]) -> None:
        """Compute episode completion statistics for dual policy.

        Args:
            env: MettaGrid environment
            episode_rewards: Rewards for all agents
            infos: Info dictionary to update with stats
        """
        if not self.enabled or not self._agent_groups:
            return

        # Get per-agent stats from environment (not used presently)
        env.get_episode_stats()

        training_rewards = []
        npc_rewards = []
        all_rewards = []

        # Collect metrics for each agent based on their group
        for agent_id, group in self._agent_groups.items():
            if agent_id < len(episode_rewards):
                reward = episode_rewards[agent_id]
                all_rewards.append(reward)

                if group == 0:  # Training group
                    training_rewards.append(reward)
                else:  # NPC group
                    npc_rewards.append(reward)

            # We intentionally suppress logging of per-agent detailed stats to reduce plot clutter

        # Structure the metrics in the desired format
        # This creates the dual_policy/trained/*, dual_policy/npc/*, etc. structure
        if "dual_policy" not in infos:
            infos["dual_policy"] = {}

        def _stat_block(rewards: list[float]) -> dict[str, float]:
            if not rewards:
                return {
                    "count": 0.0,
                    "reward_total": 0.0,
                    "reward_mean": 0.0,
                    "reward_min": 0.0,
                    "reward_max": 0.0,
                    "reward_std": 0.0,
                    "reward_median": 0.0,
                }
            arr = np.array(rewards, dtype=np.float32)
            return {
                "count": float(len(arr)),
                "reward_total": float(np.sum(arr)),
                "reward_mean": float(np.mean(arr)),
                "reward_min": float(np.min(arr)),
                "reward_max": float(np.max(arr)),
                "reward_std": float(np.std(arr)),
                "reward_median": float(np.median(arr)),
            }

        # Exactly seven metrics per category to keep plotting concise (5 categories × 7 = 35)
        infos["dual_policy/trained"] = _stat_block(training_rewards)
        infos["dual_policy/npc"] = _stat_block(npc_rewards)
        infos["dual_policy/combined"] = _stat_block(all_rewards)
        # Map training→policy_a and npc→policy_b for consistency
        infos["dual_policy/policy_a"] = _stat_block(training_rewards)
        infos["dual_policy/policy_b"] = _stat_block(npc_rewards)

        # Keep backward compatibility with simple dual_policy metrics
        infos["dual_policy"].update(
            {
                "training_agents_count": len(training_rewards),
                "npc_agents_count": len(npc_rewards),
                "training_mean_reward": float(np.mean(training_rewards)) if training_rewards else 0.0,
                "npc_mean_reward": float(np.mean(npc_rewards)) if npc_rewards else 0.0,
                "training_total_reward": float(np.sum(training_rewards)) if training_rewards else 0.0,
                "npc_total_reward": float(np.sum(npc_rewards)) if npc_rewards else 0.0,
            }
        )

    @staticmethod
    def aggregate_distributed_stats(stats: Dict[str, list], device: torch.device) -> None:
        """Aggregate dual policy stats across distributed nodes.

        Args:
            stats: Statistics dictionary to aggregate
            device: Device for tensor operations
        """
        if not torch.distributed.is_initialized():
            return

        # Check if we have dual policy stats to aggregate
        if "dual_policy" not in stats or not stats["dual_policy"]:
            return

        # Metrics to aggregate - including new structured metrics
        metrics_to_aggregate = [
            "training_mean_reward",
            "npc_mean_reward",
            "training_total_reward",
            "npc_total_reward",
            "training_agents_count",
            "npc_agents_count",
        ]

        # Also aggregate the new structured metrics
        structured_keys = [
            "dual_policy/trained",
            "dual_policy/npc",
            "dual_policy/combined",
            "dual_policy/policy_a",
            "dual_policy/policy_b",
        ]

        # Aggregate basic dual_policy metrics
        for metric in metrics_to_aggregate:
            values = []
            for episode_stats in stats.get("dual_policy", []):
                if isinstance(episode_stats, dict) and metric in episode_stats:
                    values.append(episode_stats[metric])

            if values:
                # Create tensor and all_reduce
                tensor = torch.tensor(values, dtype=torch.float32, device=device)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                world_size = torch.distributed.get_world_size()

                # Update stats with aggregated values
                for i, episode_stats in enumerate(stats.get("dual_policy", [])):
                    if isinstance(episode_stats, dict) and i < len(values):
                        if "mean" in metric:
                            # Average across nodes for mean metrics
                            episode_stats[metric] = tensor[i].item() / world_size
                        else:
                            # Sum across nodes for count/total metrics
                            episode_stats[metric] = tensor[i].item()

        # Aggregate structured metrics
        for key in structured_keys:
            if key not in stats:
                continue

            for episode_stats in stats.get(key, []):
                if not isinstance(episode_stats, dict):
                    continue

                for metric_name, metric_value in episode_stats.items():
                    if isinstance(metric_value, (int, float)):
                        # Create tensor for this metric across all episodes
                        tensor = torch.tensor([metric_value], dtype=torch.float32, device=device)
                        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                        world_size = torch.distributed.get_world_size()

                        # Update with aggregated value
                        if "mean" in metric_name or "avg" in metric_name:
                            episode_stats[metric_name] = tensor[0].item() / world_size
                        else:
                            episode_stats[metric_name] = tensor[0].item()

    def run_dual_policy_rollout(
        self,
        training_policy: PolicyAgent,
        td: Any,  # TensorDict
        training_env_id: slice,
    ) -> None:
        """Execute rollout with both training and NPC policies.

        Args:
            training_policy: The policy being trained
            td: TensorDict containing observations and to store actions
            training_env_id: Slice indicating which environments are active
        """
        if not self.enabled or not self.npc_policy:
            # Single policy path
            with torch.no_grad():
                training_policy(td)
            return

        # Ensure we have split counts
        if self._num_agents is None or self._num_training is None or self._num_npc is None:
            # Try to infer from actions shape after running policy once
            with torch.no_grad():
                training_policy(td)
            actions = td.get("actions")
            if actions is None or actions.dim() != 3:
                # Fallback: cannot split reliably; leave as training actions
                return
            _, num_agents, _ = actions.shape
            self.initialize_agent_split(num_agents)

        num_npc = int(self._num_npc or 0)
        if num_npc <= 0:
            with torch.no_grad():
                training_policy(td)
            return

        # Run both policies
        with torch.no_grad():
            training_policy(td)
            training_actions = td["actions"].clone()
            npc_td = td.clone()
            self.npc_policy(npc_td)
            npc_actions = npc_td["actions"]

        # Combine actions by contiguous slice: first num_npc are NPC
        if training_actions.dim() == 3 and npc_actions.dim() == 3:
            td["actions"][:, :num_npc, :] = npc_actions[:, :num_npc, :]
        else:
            # Unsupported shape; leave actions as produced by training policy
            return

        # Optionally combine values
        if "values" in td and "values" in npc_td:
            tr_vals = td["values"]
            npc_vals = npc_td["values"]
            # Reshape if flattened
            if tr_vals.dim() == 1 and tr_vals.numel() == training_actions.shape[0] * training_actions.shape[1]:
                B, A = training_actions.shape[0], training_actions.shape[1]
                tr_vals = tr_vals.view(B, A)
                npc_vals = npc_vals.view(B, A)
                tr_flat = True
            else:
                tr_flat = False
            if tr_vals.dim() == 2:
                tr_vals[:, :num_npc] = npc_vals[:, :num_npc]
                td["values"] = tr_vals.flatten() if tr_flat else tr_vals

        # Optionally combine log-probs (use act_log_prob key)
        if "act_log_prob" in td and "act_log_prob" in npc_td:
            tr_lp = td["act_log_prob"]
            npc_lp = npc_td["act_log_prob"]
            if tr_lp.dim() == 1 and tr_lp.numel() == training_actions.shape[0] * training_actions.shape[1]:
                B, A = training_actions.shape[0], training_actions.shape[1]
                tr_lp = tr_lp.view(B, A)
                npc_lp = npc_lp.view(B, A)
                lp_flat = True
            else:
                lp_flat = False
            if tr_lp.dim() == 2:
                tr_lp[:, :num_npc] = npc_lp[:, :num_npc]
                td["act_log_prob"] = tr_lp.flatten() if lp_flat else tr_lp
