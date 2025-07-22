"""Multi-agent trainer coordinating multiple policies."""

from typing import Dict, List, Optional, Any
import torch
from torch import Tensor

from metta.rl.trainer import MettaTrainer
from metta.rl.experience import Experience
from .reward import RewardStrategy, Competition
from .evolution import EvolutionStrategy, Stable
from .diversity import DiversityMetric


class MultiAgentTrainer:
    """Trains N policies in M-agent environments where M >= N."""
    
    def __init__(
        self,
        num_policies: int,
        base_config: Any,  # Will be trainer config
        reward_strategy: RewardStrategy = None,
        evolution_strategy: EvolutionStrategy = None,
        diversity_metric: Optional[DiversityMetric] = None,
        diversity_weight: float = 0.0,
        policy_store: Any = None,
    ):
        self.num_policies = num_policies
        self.reward_strategy = reward_strategy or Competition()
        self.evolution_strategy = evolution_strategy or Stable()
        self.diversity_metric = diversity_metric
        self.diversity_weight = diversity_weight
        self.policy_store = policy_store
        
        # Create trainers for each policy
        self.trainers: Dict[int, MettaTrainer] = {}
        self.policies: Dict[int, torch.nn.Module] = {}
        self.fitness: Dict[int, float] = {}
        self.trajectories: Dict[int, List] = {}
        
        for i in range(num_policies):
            # Each trainer gets modified config
            policy_config = self._modify_config_for_policy(base_config, i)
            trainer = self._create_trainer(policy_config, i)
            self.trainers[i] = trainer
            self.policies[i] = trainer.policy
    
    def train(self, total_timesteps: int):
        """Main training loop coordinating all policies."""
        episode = 0
        timesteps = 0
        
        while timesteps < total_timesteps:
            # Assign policies to agents
            policy_assignment = self.evolution_strategy.assign_policies(
                self._get_num_agents(), list(self.policies.keys()), episode
            )
            
            # Collect rollouts with mixed policies
            experiences = self._distributed_rollout(policy_assignment)
            
            # Distribute rewards according to strategy
            for policy_id, exp in experiences.items():
                rewards = self._compute_rewards(policy_id, exp, policy_assignment)
                exp.rewards = rewards
            
            # Train each policy on its experience
            for policy_id, trainer in self.trainers.items():
                if policy_id in experiences:
                    trainer.experience = experiences[policy_id]
                    trainer._train()
                    
                    # Track fitness
                    self.fitness[policy_id] = experiences[policy_id].rewards.mean().item()
            
            # Evolution step
            if episode % 100 == 0:  # Evolution frequency
                self.policies = self.evolution_strategy.evolve(self.fitness, self.policies)
                self._update_trainer_policies()
            
            episode += 1
            timesteps += sum(exp.batch_size for exp in experiences.values())
    
    def _distributed_rollout(self, policy_assignment: List[int]) -> Dict[int, Experience]:
        """Collect experience for all policies in shared environment."""
        # This is a sketch - actual implementation would coordinate vecenv
        experiences = {pid: [] for pid in set(policy_assignment)}
        
        # In practice, this would use a shared vecenv with policy routing
        # For now, simulate independent rollouts
        for policy_id in experiences:
            trainer = self.trainers[policy_id]
            trainer._rollout()
            experiences[policy_id] = trainer.experience
        
        return experiences
    
    def _compute_rewards(self, policy_id: int, exp: Experience, assignment: List[int]) -> Tensor:
        """Apply reward strategy and diversity bonus."""
        base_rewards = exp.rewards
        
        # Get reward distribution from strategy
        agent_ids = [i for i, pid in enumerate(assignment) if pid == policy_id]
        distributed = self.reward_strategy.distribute(base_rewards, list(range(len(assignment))), assignment)
        policy_reward = distributed[policy_id]
        
        # Add diversity bonus if configured
        if self.diversity_metric and self.diversity_weight > 0:
            diversity_bonus = self.diversity_metric.bonus(policy_id, self.policies, self.trajectories)
            policy_reward += self.diversity_weight * diversity_bonus
        
        return policy_reward
    
    def _modify_config_for_policy(self, base_config: Any, policy_id: int) -> Any:
        """Create policy-specific config variant."""
        config = base_config.copy()
        config.run = f"{config.run}_policy_{policy_id}"
        config.run_dir = f"{config.run_dir}_policy_{policy_id}"
        return config
    
    def _create_trainer(self, config: Any, policy_id: int) -> MettaTrainer:
        """Create trainer instance for a policy."""
        # This would properly instantiate MettaTrainer
        return MettaTrainer(config, wandb_run=None, policy_store=self.policy_store, 
                           sim_suite_config=None, stats_client=None)
    
    def _get_num_agents(self) -> int:
        """Get number of agents from environment."""
        return self.trainers[0].vecenv.num_agents
    
    def _update_trainer_policies(self):
        """Update trainer policies after evolution."""
        for policy_id, policy in self.policies.items():
            self.trainers[policy_id].policy = policy