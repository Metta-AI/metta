"""Reward distribution strategies for multi-agent training."""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch
from torch import Tensor


class RewardStrategy(ABC):
    """Base class for reward distribution strategies."""
    
    @abstractmethod
    def distribute(self, rewards: Tensor, agent_ids: List[int], policy_ids: List[int]) -> Dict[int, Tensor]:
        """Distribute rewards to policies based on strategy.
        
        Args:
            rewards: (num_agents,) tensor of individual rewards
            agent_ids: List mapping position to agent ID
            policy_ids: List mapping agent ID to policy ID
            
        Returns:
            Dict mapping policy ID to aggregated reward tensor
        """
        pass


class Competition(RewardStrategy):
    """Individual rewards - each policy optimizes its own agents' rewards."""
    
    def distribute(self, rewards: Tensor, agent_ids: List[int], policy_ids: List[int]) -> Dict[int, Tensor]:
        policy_rewards = {}
        for i, agent_id in enumerate(agent_ids):
            policy_id = policy_ids[agent_id]
            if policy_id not in policy_rewards:
                policy_rewards[policy_id] = []
            policy_rewards[policy_id].append(rewards[i])
        
        return {pid: torch.stack(r).mean() for pid, r in policy_rewards.items()}


class Collaboration(RewardStrategy):
    """Shared rewards - all policies optimize collective reward."""
    
    def __init__(self, sharing_fn=torch.mean):
        self.sharing_fn = sharing_fn
    
    def distribute(self, rewards: Tensor, agent_ids: List[int], policy_ids: List[int]) -> Dict[int, Tensor]:
        shared_reward = self.sharing_fn(rewards)
        unique_policies = set(policy_ids[aid] for aid in agent_ids)
        return {pid: shared_reward for pid in unique_policies}