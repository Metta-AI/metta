"""Muesli training loop implementation."""

import logging
from typing import Dict, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from metta.rl.muesli.agent import MuesliAgent
from metta.rl.muesli.config import MuesliConfig
from metta.rl.muesli.losses import compute_muesli_losses
from metta.rl.muesli.replay_buffer import MuesliReplayBuffer
from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.agent.policy_state import PolicyState

logger = logging.getLogger(__name__)


def collect_trajectories(
    agent: MuesliAgent,
    envs: Any,
    num_steps: int,
    device: torch.device,
    replay_buffer: MuesliReplayBuffer
) -> List[Dict[str, Tensor]]:
    """Collect trajectories using the current policy.
    
    Args:
        agent: Muesli agent
        envs: Vectorized environments
        num_steps: Number of steps to collect per environment
        device: Device to use
        replay_buffer: Replay buffer to add trajectories to
        
    Returns:
        List of trajectory dictionaries
    """
    num_envs = envs.num_envs
    trajectories = []
    
    # Initialize
    obs = envs.reset()
    lstm_state = None
    
    # Storage for current episodes
    episode_data = [[] for _ in range(num_envs)]
    
    for step in range(num_steps):
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        
        # Get action and value from agent
        with torch.no_grad():
            output = agent(obs_tensor, lstm_state)
            actions = output['action']
            values = output['value']
            policies = torch.softmax(output['policy_logits'], dim=-1)
            lstm_state = output['lstm_state']
            
        # Step environment
        next_obs, rewards, dones, infos = envs.step(actions.cpu().numpy())
        
        # Store step data
        for i in range(num_envs):
            episode_data[i].append({
                'obs': obs_tensor[i],
                'action': actions[i],
                'reward': rewards[i],
                'value': values[i].item(),
                'policy': policies[i],
                'done': dones[i],
                'info': infos[i]
            })
            
            # If episode ended, add to replay buffer
            if dones[i]:
                # Add trajectory to replay buffer
                for step_data in episode_data[i]:
                    replay_buffer.add_step(
                        obs=step_data['obs'],
                        action=step_data['action'],
                        reward=step_data['reward'],
                        next_obs=torch.tensor(next_obs[i], device=device, dtype=torch.float32),
                        done=step_data['done'],
                        value=step_data['value'],
                        policy=step_data['policy'],
                        info=step_data['info']
                    )
                
                # Reset episode storage
                episode_data[i] = []
                
        obs = next_obs
        
    # Convert remaining episodes to trajectories
    for i in range(num_envs):
        if len(episode_data[i]) > 0:
            trajectory = {
                'obs': torch.stack([s['obs'] for s in episode_data[i]]),
                'actions': torch.stack([s['action'] for s in episode_data[i]]),
                'rewards': torch.tensor([s['reward'] for s in episode_data[i]], device=device),
                'values': torch.tensor([s['value'] for s in episode_data[i]], device=device),
                'policies': torch.stack([s['policy'] for s in episode_data[i]]),
                'dones': torch.tensor([s['done'] for s in episode_data[i]], device=device),
            }
            trajectories.append(trajectory)
            
    return trajectories


def muesli_train(
    agent: MuesliAgent,
    envs: Any,
    config: MuesliConfig,
    optimizer: optim.Optimizer,
    replay_buffer: MuesliReplayBuffer,
    experience: Experience,
    losses: Losses,
    training_steps: int,
    agent_step: int,
    epoch: int,
    device: torch.device,
    wandb_run: Optional[Any] = None
) -> Dict[str, float]:
    """Main Muesli training function.
    
    Args:
        agent: Muesli agent
        envs: Vectorized environments
        config: Muesli configuration
        optimizer: Optimizer
        replay_buffer: Replay buffer
        experience: Experience buffer for fresh data
        losses: Loss tracker
        training_steps: Number of training steps to perform
        agent_step: Current agent step count
        epoch: Current epoch
        device: Device to use
        wandb_run: Optional wandb run for logging
        
    Returns:
        Dictionary of training metrics
    """
    losses.zero()
    metrics = {}
    
    # Collect fresh trajectories
    logger.info(f"Collecting fresh trajectories...")
    fresh_trajectories = collect_trajectories(
        agent, envs, experience.batch_size // envs.num_envs, device, replay_buffer
    )
    
    # Process fresh data into experience buffer format
    if len(fresh_trajectories) > 0:
        # Concatenate all fresh trajectories
        fresh_batch = {
            'obs': torch.cat([t['obs'] for t in fresh_trajectories], dim=0),
            'actions': torch.cat([t['actions'] for t in fresh_trajectories], dim=0),
            'rewards': torch.cat([t['rewards'] for t in fresh_trajectories], dim=0),
            'values': torch.cat([t['values'] for t in fresh_trajectories], dim=0),
            'policies': torch.cat([t['policies'] for t in fresh_trajectories], dim=0),
            'dones': torch.cat([t['dones'] for t in fresh_trajectories], dim=0),
        }
    
    # Training loop
    for step in range(training_steps):
        # Sample batch with mixed on-policy/off-policy data
        if config.mixed_training and replay_buffer.is_ready(config.replay.capacity // 10):
            # Sample from replay buffer
            replay_size = int(config.replay.replay_fraction * config.minibatch_size)
            fresh_size = config.minibatch_size - replay_size
            
            # Get replay batch
            replay_batch = replay_buffer.sample(
                replay_size,
                alpha=config.replay.priority_alpha,
                beta=config.replay.priority_beta
            )
            
            # Get fresh batch (sample from recent trajectories)
            if len(fresh_trajectories) > 0 and fresh_size > 0:
                # Sample indices from fresh data
                fresh_indices = np.random.choice(
                    len(fresh_batch['obs']), 
                    size=min(fresh_size, len(fresh_batch['obs'])),
                    replace=False
                )
                
                # Create mini fresh batch
                mini_fresh_batch = {
                    'obs': fresh_batch['obs'][fresh_indices],
                    'actions': fresh_batch['actions'][fresh_indices],
                    'rewards': fresh_batch['rewards'][fresh_indices],
                    'values': fresh_batch['values'][fresh_indices],
                    'policies': fresh_batch['policies'][fresh_indices],
                    'dones': fresh_batch['dones'][fresh_indices],
                }
                
                # Combine batches
                # Note: This is simplified - in practice you'd need to handle
                # sequences properly for model learning
                combined_batch = replay_batch
            else:
                combined_batch = replay_batch
        else:
            # Use only fresh data
            if len(fresh_trajectories) == 0:
                logger.warning("No fresh trajectories available for training")
                continue
                
            # Create batch from fresh data
            batch_indices = np.random.choice(
                len(fresh_batch['obs']),
                size=min(config.minibatch_size, len(fresh_batch['obs'])),
                replace=False
            )
            
            combined_batch = {
                'obs': fresh_batch['obs'][batch_indices].unsqueeze(1),  # Add sequence dim
                'actions': fresh_batch['actions'][batch_indices].unsqueeze(1),
                'rewards': fresh_batch['rewards'][batch_indices].unsqueeze(1),
                'values': fresh_batch['values'][batch_indices].unsqueeze(1),
                'policies': fresh_batch['policies'][batch_indices].unsqueeze(1),
                'dones': fresh_batch['dones'][batch_indices].unsqueeze(1),
                'mask': torch.ones_like(fresh_batch['dones'][batch_indices].unsqueeze(1), dtype=torch.bool),
                'weights': torch.ones(len(batch_indices), device=device),
            }
            
        # Compute Retrace targets if we have sequences
        if 'mask' in combined_batch:
            with torch.no_grad():
                # Get target values for all timesteps
                batch_size, seq_len = combined_batch['actions'].shape
                target_values = []
                target_policies = []
                
                for t in range(seq_len + 1):  # +1 for bootstrap value
                    if t < seq_len:
                        obs_t = combined_batch['obs'][:, t]
                    else:
                        # Last observation for bootstrap
                        obs_t = combined_batch['obs'][:, -1]
                        
                    hidden_t, _ = agent.target_network.representation(obs_t)
                    policy_logits_t, value_logits_t = agent.target_network.prediction(hidden_t)
                    
                    value_probs_t = torch.softmax(value_logits_t, dim=-1)
                    value_t = support_to_scalar(
                        value_probs_t,
                        config.categorical.support_size,
                        config.categorical.value_min,
                        config.categorical.value_max
                    )
                    target_values.append(value_t)
                    
                    if t < seq_len:
                        policy_probs_t = torch.softmax(policy_logits_t, dim=-1)
                        target_policies.append(policy_probs_t)
                        
                target_values = torch.stack(target_values, dim=1)
                target_policies = torch.stack(target_policies, dim=1)
                
                # Compute Retrace targets
                retrace_targets = replay_buffer.compute_retrace_targets(
                    combined_batch,
                    target_values,
                    target_policies,
                    lambda_=config.retrace.lambda_,
                    rho_max=config.retrace.rho_max
                )
                
                combined_batch['retrace_targets'] = retrace_targets
                
        # Compute losses
        batch_losses, batch_metrics = compute_muesli_losses(
            agent,
            agent.target_network,
            combined_batch,
            config,
            agent_step + step
        )
        
        # Backward pass
        optimizer.zero_grad()
        batch_losses['total'].backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            agent.parameters(), 
            config.max_grad_norm
        )
        
        optimizer.step()
        
        # Update metrics
        for key, value in batch_losses.items():
            if key != 'total':
                losses.__dict__[f"{key}_sum"] = losses.__dict__.get(f"{key}_sum", 0) + value.item()
                
        for key, value in batch_metrics.items():
            metrics[key] = value
            
        losses.minibatches_processed += 1
        
        # Update priorities if using prioritized replay
        if config.replay.priority_alpha > 0 and 'indices' in combined_batch:
            # Compute TD errors for priority updates
            with torch.no_grad():
                td_errors = torch.abs(
                    combined_batch['rewards'][:, 0] + 
                    replay_buffer.gamma * combined_batch['retrace_targets'][:, 0] -
                    combined_batch['values'][:, 0]
                )
                replay_buffer.update_priorities(
                    combined_batch['indices'],
                    td_errors
                )
                
        # Update target network
        if step % config.target_network.update_freq == 0:
            agent.update_target_network(config.target_network.tau)
            
        # Update variance estimate
        if 'advantages' in batch_metrics:
            advantages = batch_metrics.get('advantages_tensor', 
                                         torch.tensor([batch_metrics['advantages_mean']]))
            agent.update_variance_estimate(
                advantages, 
                config.cmpo.variance_decay
            )
            
    # Log metrics
    if wandb_run:
        log_dict = {
            f"muesli/{key}": value 
            for key, value in losses.stats().items()
        }
        log_dict.update({
            f"muesli/{key}": value
            for key, value in metrics.items()
        })
        log_dict['muesli/grad_norm'] = grad_norm.item()
        log_dict['muesli/replay_buffer_size'] = len(replay_buffer)
        log_dict['muesli/variance_estimate'] = agent.variance_estimate.item()
        
        wandb_run.log(log_dict, step=agent_step)
        
    return metrics


# Helper function to support scalar conversion (if not imported)
def support_to_scalar(
    categorical: Tensor,
    support_size: int,
    value_min: float,
    value_max: float
) -> Tensor:
    """Convert categorical distribution to scalar value."""
    support = torch.linspace(value_min, value_max, support_size, device=categorical.device)
    return (categorical * support).sum(dim=-1)