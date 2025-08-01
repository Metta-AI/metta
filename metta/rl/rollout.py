"""Rollout phase functions for Metta training."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_record import PolicyRecord
from metta.common.profiling.stopwatch import Stopwatch
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


def get_observation(
    vecenv: Any,  # pufferlib VecEnv instance
    device: torch.device,
    timer: Stopwatch,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, list, slice, Tensor, int]:
    """Get observations from vectorized environment and convert to tensors."""
    with timer("_rollout.env"):
        o, r, d, t, info, env_id, mask = vecenv.recv()

    training_env_id = slice(env_id[0], env_id[-1] + 1)

    mask = torch.as_tensor(mask)
    num_steps = int(mask.sum().item())

    # Convert to tensors
    o = torch.as_tensor(o).to(device, non_blocking=True)
    r = torch.as_tensor(r).to(device, non_blocking=True)
    d = torch.as_tensor(d).to(device, non_blocking=True)
    t = torch.as_tensor(t).to(device, non_blocking=True)

    return o, r, d, t, info, training_env_id, mask, num_steps


def send_observation(
    vecenv: Any,
    actions: Tensor,
    dtype_actions: Any,
    timer: Any,
) -> None:
    """Send actions back to the vectorized environment."""
    with timer("_rollout.env"):
        vecenv.send(actions.cpu().numpy().astype(dtype_actions))


def run_dual_policy_rollout(
    training_policy: torch.nn.Module,
    npc_policy_record: PolicyRecord,
    observations: Tensor,
    experience: Experience,
    training_env_id_start: int,
    device: torch.device,
    training_agents_pct: float,
    num_agents_per_env: int,
    num_envs: int,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
    """Run dual-policy rollout where some agents use training policy and others use NPC policy.

    Args:
        training_policy: The policy being trained
        npc_policy_record: The NPC policy record loaded from wandb URI
        observations: Observations tensor of shape (total_agents, *obs_shape)
        experience: Experience buffer
        training_env_id_start: Starting environment ID
        device: Device to run inference on
        training_agents_pct: Percentage of agents that use training policy
        num_agents_per_env: Number of agents per environment
        num_envs: Number of environments

    Returns:
        Tuple of (actions, selected_action_log_probs, values, lstm_state_to_store)
        Only includes data from training policy agents, NPC data is excluded
    """
    with torch.no_grad():
        # Calculate agent indices for training vs NPC policies
        training_agents_per_env = max(1, int(num_agents_per_env * training_agents_pct))
        npc_agents_per_env = num_agents_per_env - training_agents_per_env

        # Create index matrices for all environments
        total_agents = num_envs * num_agents_per_env
        idx_matrix = torch.arange(total_agents, device=device).reshape(num_envs, num_agents_per_env)

        # Training policy agents: first training_agents_per_env agents in each env
        training_idxs = idx_matrix[:, :training_agents_per_env].reshape(-1)
        # NPC agents: remaining agents in each env
        npc_idxs = (
            idx_matrix[:, training_agents_per_env:].reshape(-1)
            if npc_agents_per_env > 0
            else torch.tensor([], device=device, dtype=torch.long)
        )

        # Get observations for training policy agents
        training_obs = observations[training_idxs]
        npc_obs = (
            observations[npc_idxs] if len(npc_idxs) > 0 else torch.empty(0, *observations.shape[1:], device=device)
        )

        # Run inference for training policy
        training_actions, training_log_probs, training_values, training_lstm_state = run_policy_inference(
            training_policy, training_obs, experience, training_env_id_start, device
        )

        # Run inference for NPC policy if there are NPC agents
        if len(npc_idxs) > 0:
            npc_policy = npc_policy_record.policy
            npc_actions, npc_log_probs, npc_values, npc_lstm_state = run_policy_inference(
                npc_policy, npc_obs, experience, training_env_id_start, device
            )
        else:
            # No NPC agents, create empty tensors
            npc_actions = torch.empty(0, device=device, dtype=torch.long)

        # Stitch actions back together in original order
        all_actions = torch.zeros(total_agents, device=device, dtype=torch.long)
        all_actions[training_idxs] = training_actions
        if len(npc_idxs) > 0:
            all_actions[npc_idxs] = npc_actions

        # Return only training policy data for experience storage
        return all_actions, training_log_probs, training_values, training_lstm_state


def run_policy_inference(
    policy: PolicyAgent,
    observations: Tensor,
    experience: Experience,
    training_env_id_start: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
    """Run policy inference and return actions, log probs, values, and LSTM state."""
    with torch.no_grad():
        # Get policy outputs
        policy_outputs = policy(observations)

        # Extract actions and log probabilities
        actions = policy_outputs.actions
        selected_action_log_probs = policy_outputs.selected_action_log_probs
        values = policy_outputs.values

        # Get LSTM state if available
        lstm_state_to_store = None
        if hasattr(policy_outputs, "lstm_state") and policy_outputs.lstm_state is not None:
            lstm_state_to_store = {
                "h": policy_outputs.lstm_state[0].detach(),
                "c": policy_outputs.lstm_state[1].detach(),
            }

        return actions, selected_action_log_probs, values, lstm_state_to_store


def get_lstm_config(policy: PolicyAgent) -> Tuple[int, int]:
    """Get LSTM configuration from policy."""
    # Check if policy has LSTM layers
    if hasattr(policy, "_core_") and "_core_" in policy._modules:
        core_module = policy._core_
        if hasattr(core_module, "_net") and hasattr(core_module._net, "num_layers"):
            return core_module._net.hidden_size, core_module._net.num_layers

    # Default values if no LSTM found
    return 256, 1
