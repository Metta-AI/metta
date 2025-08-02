"""Muesli replay buffer with multi-step storage and Retrace support."""

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor


class Trajectory:
    """Container for a single trajectory."""
    def __init__(
        self,
        obs: List[Tensor],
        actions: List[Tensor],
        rewards: List[float],
        values: List[float],
        policies: List[Tensor],
        dones: List[bool],
        infos: Optional[List[Dict]] = None
    ):
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.policies = policies
        self.dones = dones
        self.infos = infos or [{} for _ in range(len(obs))]
        
    def __len__(self):
        return len(self.obs)
        
    def to_tensor(self, device: torch.device) -> Dict[str, Tensor]:
        """Convert trajectory to tensors."""
        return {
            'obs': torch.stack(self.obs).to(device),
            'actions': torch.stack(self.actions).to(device),
            'rewards': torch.tensor(self.rewards, device=device),
            'values': torch.tensor(self.values, device=device),
            'policies': torch.stack(self.policies).to(device),
            'dones': torch.tensor(self.dones, device=device, dtype=torch.float32),
        }


class MuesliReplayBuffer:
    """Replay buffer for Muesli with multi-step unrolling support.
    
    Stores trajectories and supports:
    - Multi-step sequence sampling for model learning
    - Retrace target computation
    - Mixed on-policy/off-policy sampling
    - Priority-based sampling (optional)
    """
    
    def __init__(
        self,
        capacity: int,
        unroll_length: int = 5,
        gamma: float = 0.997,
        device: torch.device = None
    ):
        self.capacity = capacity
        self.unroll_length = unroll_length
        self.gamma = gamma
        self.device = device or torch.device('cpu')
        
        # Storage
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
        # Episode tracking
        self.current_episode = []
        self.total_steps = 0
        
    def add_step(
        self,
        obs: Tensor,
        action: Tensor,
        reward: float,
        next_obs: Tensor,
        done: bool,
        value: float,
        policy: Tensor,
        info: Optional[Dict] = None
    ):
        """Add a single step to the current episode."""
        self.current_episode.append({
            'obs': obs.cpu(),
            'action': action.cpu(),
            'reward': reward,
            'next_obs': next_obs.cpu(),
            'done': done,
            'value': value,
            'policy': policy.cpu(),
            'info': info or {}
        })
        
        if done:
            self._finalize_episode()
            
    def _finalize_episode(self):
        """Process and store the completed episode."""
        if not self.current_episode:
            return
            
        # Extract data from episode
        obs = [step['obs'] for step in self.current_episode]
        actions = [step['action'] for step in self.current_episode]
        rewards = [step['reward'] for step in self.current_episode]
        values = [step['value'] for step in self.current_episode]
        policies = [step['policy'] for step in self.current_episode]
        dones = [step['done'] for step in self.current_episode]
        infos = [step['info'] for step in self.current_episode]
        
        # Add final observation
        obs.append(self.current_episode[-1]['next_obs'])
        
        # Create trajectory
        trajectory = Trajectory(obs, actions, rewards, values, policies, dones, infos)
        
        # Store multi-step sequences
        T = len(trajectory)
        for t in range(T - self.unroll_length + 1):
            # Extract sequence starting at time t
            end_t = min(t + self.unroll_length, T)
            
            sequence = {
                'start_idx': t,
                'length': end_t - t,
                'obs': obs[t:end_t+1],  # Include next obs for bootstrapping
                'actions': actions[t:end_t],
                'rewards': rewards[t:end_t],
                'values': values[t:end_t],
                'policies': policies[t:end_t],
                'dones': dones[t:end_t],
            }
            
            # Compute initial priority (can be based on TD error)
            priority = self._compute_priority(sequence)
            
            self.buffer.append(sequence)
            self.priorities.append(priority)
            self.total_steps += 1
            
        # Clear episode buffer
        self.current_episode = []
        
    def _compute_priority(self, sequence: Dict) -> float:
        """Compute priority for a sequence (for prioritized replay)."""
        # Simple priority based on average reward
        # Can be replaced with TD error or other metrics
        rewards = sequence['rewards']
        return max(abs(sum(rewards)), 1.0)
        
    def sample(
        self, 
        batch_size: int,
        alpha: float = 0.0,
        beta: float = 0.6
    ) -> Dict[str, Tensor]:
        """Sample a batch of sequences from the buffer.
        
        Args:
            batch_size: Number of sequences to sample
            alpha: Priority exponent (0 = uniform sampling)
            beta: Importance sampling correction
            
        Returns:
            Batch dictionary with multi-step sequences
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
            
        # Compute sampling probabilities
        if alpha > 0:
            priorities = np.array(self.priorities)
            probs = priorities ** alpha
            probs /= probs.sum()
        else:
            probs = np.ones(len(self.buffer)) / len(self.buffer)
            
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Compute importance sampling weights
        if alpha > 0:
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()
        else:
            weights = np.ones(batch_size)
            
        # Collect sequences
        batch_sequences = []
        for idx in indices:
            batch_sequences.append(self.buffer[idx])
            
        # Convert to tensors
        batch = self._sequences_to_batch(batch_sequences)
        batch['weights'] = torch.tensor(weights, device=self.device, dtype=torch.float32)
        batch['indices'] = torch.tensor(indices, device=self.device, dtype=torch.long)
        
        return batch
        
    def _sequences_to_batch(self, sequences: List[Dict]) -> Dict[str, Tensor]:
        """Convert list of sequences to batched tensors."""
        batch_size = len(sequences)
        max_length = max(seq['length'] for seq in sequences)
        
        # Initialize tensors with padding
        obs_shape = sequences[0]['obs'][0].shape
        action_shape = sequences[0]['actions'][0].shape
        policy_shape = sequences[0]['policies'][0].shape
        
        batch_obs = torch.zeros(batch_size, max_length + 1, *obs_shape, device=self.device)
        batch_actions = torch.zeros(batch_size, max_length, *action_shape, device=self.device)
        batch_rewards = torch.zeros(batch_size, max_length, device=self.device)
        batch_values = torch.zeros(batch_size, max_length, device=self.device)
        batch_policies = torch.zeros(batch_size, max_length, *policy_shape, device=self.device)
        batch_dones = torch.zeros(batch_size, max_length, device=self.device)
        batch_mask = torch.zeros(batch_size, max_length, device=self.device, dtype=torch.bool)
        
        # Fill tensors
        for i, seq in enumerate(sequences):
            length = seq['length']
            
            # Stack observations (including next obs)
            obs_tensor = torch.stack(seq['obs']).to(self.device)
            batch_obs[i, :length+1] = obs_tensor
            
            # Stack other data
            batch_actions[i, :length] = torch.stack(seq['actions']).to(self.device)
            batch_rewards[i, :length] = torch.tensor(seq['rewards'], device=self.device)
            batch_values[i, :length] = torch.tensor(seq['values'], device=self.device)
            batch_policies[i, :length] = torch.stack(seq['policies']).to(self.device)
            batch_dones[i, :length] = torch.tensor(seq['dones'], device=self.device, dtype=torch.float32)
            batch_mask[i, :length] = True
            
        return {
            'obs': batch_obs,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'values': batch_values,
            'policies': batch_policies,
            'dones': batch_dones,
            'mask': batch_mask,
        }
        
    def compute_retrace_targets(
        self,
        batch: Dict[str, Tensor],
        target_values: Tensor,
        target_policies: Tensor,
        lambda_: float = 0.95,
        rho_max: float = 1.0
    ) -> Tensor:
        """Compute Retrace targets for value learning.
        
        Args:
            batch: Batch of sequences from replay buffer
            target_values: Values from target network [batch, seq_len+1]
            target_policies: Policies from target network [batch, seq_len, num_actions]
            lambda_: Retrace lambda parameter
            rho_max: Maximum importance sampling ratio
            
        Returns:
            Retrace targets [batch, seq_len]
        """
        batch_size, seq_len = batch['rewards'].shape
        device = batch['rewards'].device
        
        # Extract data
        rewards = batch['rewards']
        dones = batch['dones']
        mask = batch['mask']
        actions = batch['actions']
        behavior_policies = batch['policies']
        
        # Compute importance sampling ratios
        if len(target_policies.shape) == 3:  # Discrete actions
            # Get probabilities for taken actions
            action_indices = actions.long()
            target_probs = torch.gather(target_policies, dim=-1, index=action_indices.unsqueeze(-1)).squeeze(-1)
            behavior_probs = torch.gather(behavior_policies, dim=-1, index=action_indices.unsqueeze(-1)).squeeze(-1)
        else:  # Continuous actions
            # Assume Gaussian policies, compute ratios
            # This is simplified - in practice you'd need proper distribution handling
            target_probs = target_policies
            behavior_probs = behavior_policies
            
        # Clip importance sampling ratios
        rho = torch.clamp(target_probs / (behavior_probs + 1e-8), max=rho_max)
        c = torch.clamp(rho, max=1.0)  # For traces
        
        # Initialize Retrace targets
        retrace_targets = torch.zeros_like(rewards)
        
        # Backward pass to compute targets
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                # Last step: use bootstrap value if not done
                retrace_targets[:, t] = rewards[:, t] + self.gamma * (1 - dones[:, t]) * target_values[:, t+1]
            else:
                # Compute TD error
                td_error = rewards[:, t] + self.gamma * (1 - dones[:, t]) * target_values[:, t+1] - target_values[:, t]
                
                # Retrace correction
                retrace_targets[:, t] = target_values[:, t] + c[:, t] * (
                    rewards[:, t] + self.gamma * (1 - dones[:, t]) * (
                        rho[:, t+1] * (retrace_targets[:, t+1] - target_values[:, t+1]) + 
                        target_values[:, t+1]
                    ) - target_values[:, t]
                )
                
        # Apply mask
        retrace_targets = retrace_targets * mask
        
        return retrace_targets
        
    def update_priorities(self, indices: Tensor, priorities: Tensor):
        """Update priorities for sampled sequences."""
        for idx, priority in zip(indices.cpu().numpy(), priorities.cpu().numpy()):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)
        
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough data for sampling."""
        return len(self.buffer) >= min_size