"""Rollout functions for running policy inference during training."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from pufferlib import PufferlibVecEnv

    from metta.agent.metta_agent import PolicyAgent
    from metta.agent.policy_record import PolicyRecord
    from metta.agent.policy_state import PolicyState
    from metta.rl.experience import Experience
else:
    # Runtime imports to avoid circular dependencies
    PolicyAgent = Any
    PolicyRecord = Any
    PolicyState = Any
    Experience = Any
    PufferlibVecEnv = Any

from metta.common.profiling.stopwatch import Stopwatch


def get_observation(
    vecenv: PufferlibVecEnv,
    device: torch.device,
    timer: Stopwatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list, slice, Tensor, int]:
    """Receive observations from the environment."""
    with timer("get_observation.recv"):
        o, r, d, t, info, env_id, mask = vecenv.recv()
        # Handle both slice and numpy array cases for env_id
        if hasattr(env_id, "start") and hasattr(env_id, "stop"):
            # It's already a slice
            env_id = slice(env_id.start, env_id.stop)
        elif isinstance(env_id, np.ndarray):
            # It's a numpy array, convert to slice using first and last elements
            if len(env_id) > 0:
                env_id = slice(int(env_id[0]), int(env_id[-1]) + 1)
            else:
                env_id = slice(0, 0)
        # If it's something else, try to use it as-is

    with timer("get_observation.convert"):
        o = torch.as_tensor(o, device=device, dtype=torch.uint8)
        r = torch.as_tensor(r, device=device, dtype=torch.float32)
        d = torch.as_tensor(d, device=device, dtype=torch.float32)
        t = torch.as_tensor(t, device=device, dtype=torch.float32)
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool)

    training_env_id = env_id
    num_steps = o.shape[0]

    return o, r, d, t, info, training_env_id, mask, num_steps


def send_observation(
    vecenv: PufferlibVecEnv,
    actions: Tensor,
    dtype_actions: np.dtype,
    timer: Stopwatch,
) -> None:
    """Send actions to the environment."""
    with timer("send_observation"):
        if not vecenv.async_io:
            actions = actions.cpu().numpy()
        else:
            actions = actions.cpu().numpy()

        vecenv.send(actions.astype(dtype_actions))


def run_npc_policy_inference(
    npc_policy: torch.nn.Module,
    observations: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run NPC policy inference with properly initialized LSTM states.

    NPC policies need proper LSTM states that match the batch size of their observations.
    We initialize fresh LSTM states for each inference call.

    Args:
        npc_policy: The NPC policy
        observations: Observations tensor
        device: Device to run inference on

    Returns:
        Tuple of (actions, log_probs, values) - no LSTM state needed
    """
    with torch.no_grad():
        # Dynamic import to avoid circular dependency
        try:
            from metta.agent.policy_state import PolicyState
        except ImportError:
            # Create a simple PolicyState if import fails
            class PolicyState:
                def __init__(self, lstm_h=None, lstm_c=None):
                    self.lstm_h = lstm_h
                    self.lstm_c = lstm_c

        # Get LSTM configuration from the NPC policy
        hidden_size, num_layers = get_lstm_config(npc_policy)
        batch_size = observations.shape[0]

        # Initialize fresh LSTM states for NPC policy
        # This ensures the states match the current batch size and policy configuration
        lstm_h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        lstm_c = torch.zeros(num_layers, batch_size, hidden_size, device=device)

        state = PolicyState(lstm_h=lstm_h, lstm_c=lstm_c)

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
    """Run dual-policy rollout with separate policies for students and NPCs.

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
        Tuple of (actions, log_probs, values, lstm_state) for all agents
    """
    if npc_policy_record is None or npc_policy_record.policy is None:
        # Fallback to all agents using training policy
        return run_policy_inference(
            policy=training_policy,
            observations=observations,
            experience=experience,
            training_env_id_start=training_env_id_start,
            device=device,
        )

    # Calculate agent splits
    total_agents = observations.shape[0]
    agents_per_env = num_agents_per_env
    num_student_agents_per_env = max(1, int(agents_per_env * training_agents_pct))

    # Create indices for students and NPCs (contiguous split per environment)
    student_indices = []
    npc_indices = []

    for env_idx in range(num_envs):
        env_start = env_idx * agents_per_env
        # First agents in each env are students
        student_indices.extend(range(env_start, env_start + num_student_agents_per_env))
        # Remaining agents are NPCs
        npc_indices.extend(range(env_start + num_student_agents_per_env, env_start + agents_per_env))

    # Run inference for student agents using training policy
    student_obs = observations[student_indices]
    student_actions, student_log_probs, student_values, lstm_state = run_policy_inference(
        policy=training_policy,
        observations=student_obs,
        experience=experience,
        training_env_id_start=training_env_id_start,
        device=device,
    )

    # Run inference for NPC agents using frozen NPC policy
    if len(npc_indices) > 0:
        npc_obs = observations[npc_indices]
        npc_actions, npc_log_probs, npc_values = run_npc_policy_inference(
            npc_policy=npc_policy_record.policy,
            observations=npc_obs,
            device=device,
        )

        # Combine results back in original order
        all_actions = torch.zeros(
            (total_agents, *student_actions.shape[1:]), device=device, dtype=student_actions.dtype
        )
        all_log_probs = torch.zeros(total_agents, device=device, dtype=student_log_probs.dtype)
        all_values = torch.zeros(total_agents, device=device, dtype=student_values.dtype)

        all_actions[student_indices] = student_actions
        all_actions[npc_indices] = npc_actions
        all_log_probs[student_indices] = student_log_probs
        all_log_probs[npc_indices] = npc_log_probs
        all_values[student_indices] = student_values
        all_values[npc_indices] = npc_values

        return all_actions, all_log_probs, all_values, lstm_state
    else:
        # No NPCs, return student results
        return student_actions, student_log_probs, student_values, lstm_state


def run_policy_inference(
    policy: Any,  # PolicyAgent type
    observations: Tensor,
    experience: Any,  # Experience type
    training_env_id_start: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
    """Run policy inference and return actions, log probs, values, and LSTM state."""
    with torch.no_grad():
        # Dynamic import to avoid circular dependency
        try:
            from metta.agent.policy_state import PolicyState
        except ImportError:
            # Create a simple PolicyState if import fails
            class PolicyState:
                def __init__(self, lstm_h=None, lstm_c=None):
                    self.lstm_h = lstm_h
                    self.lstm_c = lstm_c

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

        return actions, selected_action_log_probs, values.flatten(), lstm_state_to_store


def get_lstm_config(policy: torch.nn.Module) -> Tuple[int, int]:
    """Get LSTM configuration from policy."""
    # For MettaAgent, access LSTM through the lstm property
    if hasattr(policy, "lstm"):
        lstm = policy.lstm
        if hasattr(lstm, "hidden_size") and hasattr(lstm, "num_layers"):
            return int(lstm.hidden_size), int(lstm.num_layers)

    # For external policies with LSTM wrapper
    if hasattr(policy, "recurrent"):
        recurrent = policy.recurrent
        if hasattr(recurrent, "hidden_size"):
            # External policies typically use 1 layer
            return int(recurrent.hidden_size), 1

    # For policies with direct LSTM attributes
    if hasattr(policy, "hidden_size") and hasattr(policy, "num_lstm_layers"):
        return int(policy.hidden_size), int(policy.num_lstm_layers)

    # For policies with hidden_size but no explicit layers
    if hasattr(policy, "hidden_size"):
        return int(policy.hidden_size), 1

    # Default values if no LSTM found
    return 256, 1
