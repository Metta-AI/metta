"""Muesli loss functions including CMPO and model learning objectives."""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional

from metta.rl.muesli.categorical import (
    scalar_to_support, 
    support_to_scalar,
    cross_entropy_loss
)
from metta.rl.muesli.config import MuesliConfig


def compute_cmpo_policy(
    advantages: Tensor,
    prior_policy: Tensor,
    clip_bound: float = 1.0,
    epsilon: float = 1e-8
) -> Tensor:
    """Compute CMPO target policy using clipped advantages.
    
    π_CMPO(a|s) ∝ π_prior(a|s) * exp(clip(adv(s,a), -c, c))
    
    Args:
        advantages: Advantages for each action [batch, num_actions]
        prior_policy: Prior policy probabilities [batch, num_actions]
        clip_bound: Clipping bound for advantages
        epsilon: Small value for numerical stability
        
    Returns:
        CMPO target policy [batch, num_actions]
    """
    # Clip advantages
    clipped_advantages = torch.clamp(advantages, -clip_bound, clip_bound)
    
    # Compute unnormalized CMPO policy
    unnormalized = prior_policy * torch.exp(clipped_advantages)
    
    # Normalize to get valid probability distribution
    z_cmpo = unnormalized.sum(dim=-1, keepdim=True) + epsilon
    cmpo_policy = unnormalized / z_cmpo
    
    return cmpo_policy


def compute_model_based_advantages(
    hidden_states: Tensor,
    actions: Tensor,
    agent: torch.nn.Module,
    config: MuesliConfig,
    normalize: bool = True
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute model-based advantages using one-step lookahead.
    
    Args:
        hidden_states: Current hidden states [batch, hidden_size]
        actions: Available actions (for discrete) or None (for continuous)
        agent: Muesli agent (target network)
        config: Muesli configuration
        normalize: Whether to normalize advantages
        
    Returns:
        advantages: Computed advantages [batch, num_actions]
        info: Dictionary with intermediate values for logging
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    
    with torch.no_grad():
        # Get current state value
        _, current_value_logits = agent.prediction(hidden_states)
        current_value_probs = F.softmax(current_value_logits, dim=-1)
        current_values = support_to_scalar(
            current_value_probs,
            config.categorical.support_size,
            config.categorical.value_min,
            config.categorical.value_max
        )
        
        if agent.discrete_actions:
            num_actions = agent.num_actions
            
            # Compute Q-values for all actions
            q_values = torch.zeros(batch_size, num_actions, device=device)
            
            for a in range(num_actions):
                # Create action tensor
                action_batch = torch.full((batch_size,), a, device=device, dtype=torch.long)
                
                # One-step dynamics
                next_hidden, reward_logits, _ = agent.dynamics(hidden_states, action_batch)
                
                # Get next value
                _, next_value_logits = agent.prediction(next_hidden)
                next_value_probs = F.softmax(next_value_logits, dim=-1)
                next_values = support_to_scalar(
                    next_value_probs,
                    config.categorical.support_size,
                    config.categorical.value_min,
                    config.categorical.value_max
                )
                
                # Get immediate reward
                reward_probs = F.softmax(reward_logits, dim=-1)
                rewards = support_to_scalar(
                    reward_probs,
                    config.categorical.support_size,
                    config.categorical.value_min,
                    config.categorical.value_max
                )
                
                # Q(s,a) = r(s,a) + γ * V(s')
                # Note: gamma should be passed or taken from trainer config
                gamma = 0.99  # Default gamma, should be parameterized
                q_values[:, a] = rewards + gamma * next_values
                
            # Advantages = Q(s,a) - V(s)
            advantages = q_values - current_values.unsqueeze(-1)
            
        else:
            # For continuous actions, sample a set of actions
            # This is a simplified approach - in practice you might use
            # a more sophisticated method
            raise NotImplementedError("Continuous actions not yet implemented")
            
    # Normalize advantages if requested
    if normalize:
        # Use running variance estimate from agent
        std = torch.sqrt(agent.variance_estimate + 1e-8)
        advantages = advantages / std
        
    info = {
        'current_values': current_values,
        'q_values': q_values if agent.discrete_actions else None,
        'advantages_mean': advantages.mean().item(),
        'advantages_std': advantages.std().item(),
    }
    
    return advantages, info


def compute_muesli_losses(
    agent: torch.nn.Module,
    target_agent: torch.nn.Module,
    batch: Dict[str, Tensor],
    config: MuesliConfig,
    training_step: int
) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
    """Compute all Muesli losses.
    
    Args:
        agent: Current agent (being trained)
        target_agent: Target agent (EMA)
        batch: Batch of data from replay buffer
        config: Muesli configuration
        training_step: Current training step
        
    Returns:
        losses: Dictionary of loss tensors
        metrics: Dictionary of metrics for logging
    """
    device = batch['obs'].device
    batch_size, seq_len = batch['actions'].shape
    
    losses = {}
    metrics = {}
    
    # 1. Policy Gradient Loss with CMPO regularization
    # Get initial hidden states
    obs_t0 = batch['obs'][:, 0]  # [batch, obs_shape...]
    hidden_states, lstm_states = agent.representation(obs_t0)
    
    # Compute model-based advantages using target network
    advantages, adv_info = compute_model_based_advantages(
        hidden_states, None, target_agent, config, normalize=True
    )
    
    # Get current policy
    policy_logits, _ = agent.prediction(hidden_states)
    current_policy = F.softmax(policy_logits, dim=-1)
    
    # Compute CMPO target policy using target network
    with torch.no_grad():
        target_policy_logits, _ = target_agent.prediction(hidden_states)
        prior_policy = F.softmax(target_policy_logits, dim=-1)
        cmpo_policy = compute_cmpo_policy(
            advantages, prior_policy, config.cmpo.clip_bound
        )
    
    # Policy gradient loss (negative because we maximize expected advantage)
    taken_actions = batch['actions'][:, 0].long()
    action_log_probs = torch.log(current_policy + 1e-8)
    selected_log_probs = action_log_probs.gather(1, taken_actions.unsqueeze(-1)).squeeze(-1)
    
    # Weight by importance sampling if using replay
    if 'weights' in batch:
        importance_weights = batch['weights']
    else:
        importance_weights = torch.ones(batch_size, device=device)
        
    pg_loss = -(advantages.gather(1, taken_actions.unsqueeze(-1)).squeeze(-1).detach() * 
                selected_log_probs * importance_weights).mean()
    
    # CMPO regularization loss (KL divergence)
    cmpo_loss = config.cmpo.cmpo_weight * F.kl_div(
        action_log_probs,
        cmpo_policy.detach(),
        reduction='batchmean'
    )
    
    losses['policy_gradient'] = pg_loss
    losses['cmpo_regularization'] = cmpo_loss
    
    # 2. Model Learning Losses (multi-step)
    model_losses = compute_model_learning_losses(
        agent, target_agent, batch, config
    )
    losses.update(model_losses['losses'])
    metrics.update(model_losses['metrics'])
    
    # 3. Value Loss (using Retrace targets)
    value_loss = compute_value_loss(
        agent, target_agent, batch, config
    )
    losses['value'] = value_loss
    
    # Update metrics
    metrics['advantages_mean'] = adv_info['advantages_mean']
    metrics['advantages_std'] = adv_info['advantages_std']
    metrics['cmpo_max_tv_distance'] = torch.tanh(torch.tensor(config.cmpo.clip_bound / 2)).item()
    
    # Total loss
    total_loss = sum(losses.values())
    losses['total'] = total_loss
    
    return losses, metrics


def compute_model_learning_losses(
    agent: torch.nn.Module,
    target_agent: torch.nn.Module,
    batch: Dict[str, Tensor],
    config: MuesliConfig
) -> Dict[str, any]:
    """Compute model learning losses through multi-step unrolling.
    
    Args:
        agent: Current agent
        target_agent: Target agent
        batch: Batch from replay buffer
        config: Configuration
        
    Returns:
        Dictionary with losses and metrics
    """
    device = batch['obs'].device
    batch_size, seq_len = batch['actions'].shape
    mask = batch['mask']
    
    total_reward_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    
    # Initial representation
    obs_0 = batch['obs'][:, 0]
    hidden_state, lstm_state = agent.representation(obs_0)
    
    # Multi-step unrolling
    unroll_steps = min(config.model_learning.unroll_steps, seq_len)
    
    for k in range(unroll_steps):
        # Get action for this step
        action_k = batch['actions'][:, k]
        
        # Dynamics step
        next_hidden, reward_logits, lstm_state = agent.dynamics(
            hidden_state, action_k, lstm_state
        )
        
        # Prediction from next hidden state
        policy_logits, value_logits = agent.prediction(next_hidden)
        
        # Compute targets with target network
        with torch.no_grad():
            # Get target values and policies for this timestep
            if k + 1 < seq_len:
                target_obs = batch['obs'][:, k + 1]
                target_hidden, _ = target_agent.representation(target_obs)
                target_policy_logits, target_value_logits = target_agent.prediction(target_hidden)
                
                # Convert to categorical targets
                reward_target = scalar_to_support(
                    batch['rewards'][:, k],
                    config.categorical.support_size,
                    config.categorical.value_min,
                    config.categorical.value_max
                )
                
                # Use Retrace values if available, otherwise use target network values
                if 'retrace_values' in batch:
                    value_target = scalar_to_support(
                        batch['retrace_values'][:, k],
                        config.categorical.support_size,
                        config.categorical.value_min,
                        config.categorical.value_max
                    )
                else:
                    value_target = F.softmax(target_value_logits, dim=-1)
                    
                # CMPO policy as target
                # First compute advantages at this future state
                future_advantages, _ = compute_model_based_advantages(
                    target_hidden, None, target_agent, config, normalize=True
                )
                prior_policy = F.softmax(target_policy_logits, dim=-1)
                policy_target = compute_cmpo_policy(
                    future_advantages, prior_policy, config.cmpo.clip_bound
                )
            else:
                # Last step - use zeros or bootstrap
                continue
                
        # Compute losses
        step_mask = mask[:, k].float()
        
        # Reward prediction loss
        reward_loss = cross_entropy_loss(reward_logits, reward_target, reduction='none')
        reward_loss = (reward_loss * step_mask).mean()
        total_reward_loss += reward_loss
        
        # Value prediction loss
        value_loss = cross_entropy_loss(value_logits, value_target, reduction='none')
        value_loss = (value_loss * step_mask).mean()
        total_value_loss += value_loss
        
        # Policy model loss
        policy_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            policy_target,
            reduction='none'
        ).sum(dim=-1)
        policy_loss = (policy_loss * step_mask).mean()
        total_policy_loss += policy_loss
        
        # Update hidden state for next step
        hidden_state = next_hidden
        
    # Average over unroll steps
    avg_reward_loss = total_reward_loss / unroll_steps * config.model_learning.reward_weight
    avg_value_loss = total_value_loss / unroll_steps * config.model_learning.value_weight
    avg_policy_loss = total_policy_loss / unroll_steps * config.model_learning.policy_model_weight
    
    return {
        'losses': {
            'model_reward': avg_reward_loss,
            'model_value': avg_value_loss,
            'model_policy': avg_policy_loss,
        },
        'metrics': {
            'model_reward_loss': avg_reward_loss.item(),
            'model_value_loss': avg_value_loss.item(),
            'model_policy_loss': avg_policy_loss.item(),
        }
    }


def compute_value_loss(
    agent: torch.nn.Module,
    target_agent: torch.nn.Module,
    batch: Dict[str, Tensor],
    config: MuesliConfig
) -> Tensor:
    """Compute value loss using Retrace targets.
    
    Args:
        agent: Current agent
        target_agent: Target agent
        batch: Batch from replay buffer
        config: Configuration
        
    Returns:
        Value loss tensor
    """
    # Get value predictions for the trajectory
    batch_size, seq_len = batch['actions'].shape
    all_values = []
    
    # Process each timestep
    for t in range(seq_len):
        obs_t = batch['obs'][:, t]
        hidden_t, _ = agent.representation(obs_t)
        _, value_logits_t = agent.prediction(hidden_t)
        all_values.append(value_logits_t)
        
    # Stack values
    value_logits = torch.stack(all_values, dim=1)  # [batch, seq_len, support_size]
    
    # Get Retrace targets (should be computed in the training loop)
    if 'retrace_targets' in batch:
        retrace_targets = batch['retrace_targets']  # [batch, seq_len]
    else:
        # Fallback to rewards + bootstrapped values
        retrace_targets = batch['rewards']  # Simplified
        
    # Convert targets to categorical
    retrace_categorical = []
    for t in range(seq_len):
        target_cat = scalar_to_support(
            retrace_targets[:, t],
            config.categorical.support_size,
            config.categorical.value_min,
            config.categorical.value_max
        )
        retrace_categorical.append(target_cat)
        
    retrace_categorical = torch.stack(retrace_categorical, dim=1)
    
    # Compute loss
    mask = batch['mask']
    value_loss = cross_entropy_loss(
        value_logits.reshape(-1, config.categorical.support_size),
        retrace_categorical.reshape(-1, config.categorical.support_size),
        reduction='none'
    )
    value_loss = value_loss.reshape(batch_size, seq_len)
    value_loss = (value_loss * mask).sum() / mask.sum()
    
    return value_loss