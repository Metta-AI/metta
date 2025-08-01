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


def run_npc_policy_inference(
    npc_policy: torch.nn.Module,
    observations: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run NPC policy inference without interacting with experience buffer.

    NPC policies operate in open loop and don't contribute to learning.
    They don't need LSTM state management from the experience buffer.

    Args:
        npc_policy: The NPC policy
        observations: Observations tensor
        device: Device to run inference on

    Returns:
        Tuple of (actions, log_probs, values) - no LSTM state needed
    """
    with torch.no_grad():
        # Create empty policy state for NPC (no LSTM state management)
        from metta.agent.policy_state import PolicyState

        state = PolicyState(lstm_h=None, lstm_c=None)

        # Get policy outputs
        policy_outputs = npc_policy(observations, state)

        # Extract actions and log probabilities
        actions = policy_outputs[0]  # actions
        log_probs = policy_outputs[1]  # log_probs
        values = policy_outputs[2]  # values

        return actions, log_probs, values


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
        experience: Experience buffer (only used for training policy)
        training_env_id_start: Starting environment ID for training
        device: Device to run inference on
        training_agents_pct: Percentage of agents that use training policy
        num_agents_per_env: Number of agents per environment
        num_envs: Number of environments

    Returns:
        Tuple of (actions, log_probs, values, lstm_state) for training agents only
    """
    total_agents = observations.shape[0]
    num_training_agents = int(total_agents * training_agents_pct)

    # Split observations for training vs NPC agents
    training_obs = observations[:num_training_agents]
    npc_obs = observations[num_training_agents:]

    # Run training policy inference (with experience buffer interaction)
    training_actions, training_log_probs, training_values, lstm_state = run_policy_inference(
        policy=training_policy,
        observations=training_obs,
        experience=experience,
        training_env_id_start=training_env_id_start,
        device=device,
    )

    # Run NPC policy inference (without experience buffer interaction)
    npc_actions, npc_log_probs, npc_values = run_npc_policy_inference(
        npc_policy=npc_policy_record.policy,
        observations=npc_obs,
        device=device,
    )

    # Combine actions, log_probs, values (only training agents contribute to learning)
    all_actions = torch.cat([training_actions, npc_actions], dim=0)
    all_log_probs = torch.cat([training_log_probs, npc_log_probs], dim=0)
    all_values = torch.cat([training_values, npc_values], dim=0)

    # Return combined results and LSTM state from training policy only
    return all_actions, all_log_probs, all_values, lstm_state


def run_policy_inference(
    policy: PolicyAgent,
    observations: Tensor,
    experience: Experience,
    training_env_id_start: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
    """Run policy inference and return actions, log probs, values, and LSTM state."""
    with torch.no_grad():
        # Create policy state with LSTM state from experience
        from metta.agent.policy_state import PolicyState

        lstm_h, lstm_c = experience.get_lstm_state(training_env_id_start)
        state = PolicyState(lstm_h=lstm_h, lstm_c=lstm_c)

        # Get policy outputs
        policy_outputs = policy(observations, state)

        # Extract actions and log probabilities
        actions = policy_outputs[0]  # actions
        selected_action_log_probs = policy_outputs[1]  # log_probs
        values = policy_outputs[2]  # values

        # Get LSTM state if available
        lstm_state_to_store = None
        if state.lstm_h is not None and state.lstm_c is not None:
            lstm_state_to_store = {
                "lstm_h": state.lstm_h.detach(),
                "lstm_c": state.lstm_c.detach(),
            }

        return actions, selected_action_log_probs, values, lstm_state_to_store


def get_lstm_config(policy: PolicyAgent) -> Tuple[int, int]:
    """Get LSTM configuration from policy."""
    # For MettaAgent, access LSTM through the lstm property
    if hasattr(policy, "lstm"):
        lstm = policy.lstm
        if hasattr(lstm, "hidden_size") and hasattr(lstm, "num_layers"):
            return lstm.hidden_size, lstm.num_layers

    # Alternative: check if policy has direct LSTM attributes
    if hasattr(policy, "hidden_size") and hasattr(policy, "num_lstm_layers"):
        return policy.hidden_size, policy.num_lstm_layers

    # Default values if no LSTM found
    return 256, 1
