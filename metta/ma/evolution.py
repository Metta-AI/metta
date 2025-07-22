"""Evolutionary dynamics for policy populations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


class EvolutionStrategy(ABC):
    """Base class for evolutionary strategies."""
    
    @abstractmethod
    def assign_policies(self, num_agents: int, policy_ids: List[int], episode: int) -> List[int]:
        """Assign policies to agents for an episode.
        
        Returns:
            List mapping agent index to policy ID
        """
        pass
    
    @abstractmethod
    def evolve(self, fitness: Dict[int, float], policies: Dict[int, torch.nn.Module]) -> Dict[int, torch.nn.Module]:
        """Evolve population based on fitness scores."""
        pass


class Stable(EvolutionStrategy):
    """Fixed policy-agent mapping, no evolution."""
    
    def __init__(self, policy_mapping: Optional[List[int]] = None):
        self.policy_mapping = policy_mapping
    
    def assign_policies(self, num_agents: int, policy_ids: List[int], episode: int) -> List[int]:
        if self.policy_mapping is None:
            # Cycle through policies if more agents than policies
            return [policy_ids[i % len(policy_ids)] for i in range(num_agents)]
        return self.policy_mapping
    
    def evolve(self, fitness: Dict[int, float], policies: Dict[int, torch.nn.Module]) -> Dict[int, torch.nn.Module]:
        return policies  # No evolution


class Evolving(EvolutionStrategy):
    """Dynamic policy assignment with mutation and selection."""
    
    def __init__(self, mutation_rate: float = 0.01, selection_pressure: float = 2.0, 
                 crossover_rate: float = 0.1):
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.crossover_rate = crossover_rate
        self.generation = 0
    
    def assign_policies(self, num_agents: int, policy_ids: List[int], episode: int) -> List[int]:
        # Random assignment each episode
        return np.random.choice(policy_ids, size=num_agents).tolist()
    
    def evolve(self, fitness: Dict[int, float], policies: Dict[int, torch.nn.Module]) -> Dict[int, torch.nn.Module]:
        """Tournament selection + mutation."""
        self.generation += 1
        
        # Normalize fitness for selection probabilities
        fitness_vals = np.array(list(fitness.values()))
        probs = np.exp(self.selection_pressure * fitness_vals)
        probs /= probs.sum()
        
        new_policies = {}
        policy_ids = list(policies.keys())
        
        for pid in policy_ids:
            # Tournament selection
            parent_id = np.random.choice(policy_ids, p=probs)
            new_policy = self._copy_policy(policies[parent_id])
            
            # Mutation
            if np.random.rand() < self.mutation_rate:
                self._mutate_policy(new_policy)
            
            new_policies[pid] = new_policy
        
        return new_policies
    
    def _copy_policy(self, policy: torch.nn.Module) -> torch.nn.Module:
        """Deep copy a policy."""
        new_policy = type(policy)(**policy.init_kwargs)  # Assumes init_kwargs stored
        new_policy.load_state_dict(policy.state_dict())
        return new_policy
    
    def _mutate_policy(self, policy: torch.nn.Module, noise_std: float = 0.01):
        """Add Gaussian noise to policy weights."""
        with torch.no_grad():
            for param in policy.parameters():
                param.add_(torch.randn_like(param) * noise_std)