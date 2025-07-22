"""Diversity metrics and incentives for multi-agent populations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
from torch import Tensor
import numpy as np


class DiversityMetric(ABC):
    """Base class for diversity metrics."""
    
    @abstractmethod
    def compute(self, policies: Dict[int, torch.nn.Module], trajectories: Optional[Dict[int, List]] = None) -> float:
        """Compute diversity score for policy population."""
        pass
    
    @abstractmethod
    def bonus(self, policy_id: int, policies: Dict[int, torch.nn.Module], 
              trajectories: Optional[Dict[int, List]] = None) -> float:
        """Compute diversity bonus for a specific policy."""
        pass


class WeightDiversity(DiversityMetric):
    """Diversity based on policy parameter differences."""
    
    def __init__(self, metric: str = "l2"):
        self.metric = metric
    
    def compute(self, policies: Dict[int, torch.nn.Module], trajectories=None) -> float:
        """Average pairwise distance between policy weights."""
        if len(policies) < 2:
            return 0.0
        
        weights = [self._flatten_weights(p) for p in policies.values()]
        distances = []
        
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                if self.metric == "l2":
                    dist = torch.norm(weights[i] - weights[j], p=2)
                elif self.metric == "cosine":
                    dist = 1 - torch.cosine_similarity(weights[i], weights[j], dim=0)
                distances.append(dist.item())
        
        return np.mean(distances)
    
    def bonus(self, policy_id: int, policies: Dict[int, torch.nn.Module], trajectories=None) -> float:
        """Distance from policy to population centroid."""
        if len(policies) < 2:
            return 0.0
        
        target_weights = self._flatten_weights(policies[policy_id])
        other_weights = [self._flatten_weights(p) for pid, p in policies.items() if pid != policy_id]
        centroid = torch.stack(other_weights).mean(dim=0)
        
        if self.metric == "l2":
            return torch.norm(target_weights - centroid, p=2).item()
        elif self.metric == "cosine":
            return (1 - torch.cosine_similarity(target_weights, centroid, dim=0)).item()
    
    def _flatten_weights(self, policy: torch.nn.Module) -> Tensor:
        """Flatten all parameters into single vector."""
        return torch.cat([p.data.flatten() for p in policy.parameters()])


class BehavioralDiversity(DiversityMetric):
    """Diversity based on policy behavior/trajectories."""
    
    def __init__(self, state_encoder=None, action_weight: float = 0.5):
        self.state_encoder = state_encoder or (lambda x: x)  # Identity by default
        self.action_weight = action_weight
    
    def compute(self, policies: Dict[int, torch.nn.Module], trajectories: Dict[int, List]) -> float:
        """Average pairwise distance between behavioral embeddings."""
        if len(trajectories) < 2:
            return 0.0
        
        embeddings = {pid: self._compute_embedding(traj) for pid, traj in trajectories.items()}
        distances = []
        
        pids = list(embeddings.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                dist = torch.norm(embeddings[pids[i]] - embeddings[pids[j]], p=2)
                distances.append(dist.item())
        
        return np.mean(distances)
    
    def bonus(self, policy_id: int, policies: Dict[int, torch.nn.Module], 
              trajectories: Dict[int, List]) -> float:
        """Novelty of behavior compared to population."""
        if len(trajectories) < 2 or policy_id not in trajectories:
            return 0.0
        
        target_emb = self._compute_embedding(trajectories[policy_id])
        other_embs = [self._compute_embedding(traj) for pid, traj in trajectories.items() if pid != policy_id]
        
        # k-nearest neighbor distance as novelty
        k = min(3, len(other_embs))
        distances = [torch.norm(target_emb - emb, p=2).item() for emb in other_embs]
        return np.mean(sorted(distances)[:k])
    
    def _compute_embedding(self, trajectory: List[Tuple[Tensor, int]]) -> Tensor:
        """Convert trajectory to fixed-size embedding."""
        states, actions = zip(*trajectory)
        
        # Encode states
        encoded_states = torch.stack([self.state_encoder(s) for s in states])
        state_features = encoded_states.mean(dim=0)  # Temporal pooling
        
        # Action distribution
        action_counts = torch.zeros(10)  # Assume max 10 actions
        for a in actions:
            action_counts[a] += 1
        action_dist = action_counts / len(actions)
        
        # Combine features
        return torch.cat([
            state_features.flatten(),
            action_dist * self.action_weight
        ])