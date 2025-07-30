"""Rollout phase functions for Metta training."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


def get_observation(
    vecenv: Any,
    device: torch.device,
    timer: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, list, slice, Tensor, int]:
    """Get observations and other data from the vectorized environment and convert to tensors.

    Returns:
        Tuple of (observations, rewards, dones, truncations, info, training_env_id, mask, num_steps)
    """
    # Receive environment data
    with timer("_rollout.env"):
        o, r, d, t, info, env_id, mask = vecenv.recv()

    training_env_id = slice(env_id[0], env_id[-1] + 1)

    mask = torch.as_tensor(mask)
    num_steps = int(mask.sum().item())

    # Convert to tensors and batch transfer to device
    # This is more efficient than individual transfers
    o = torch.as_tensor(o)
    r = torch.as_tensor(r)
    d = torch.as_tensor(d)
    t = torch.as_tensor(t)

    if str(device) != "cpu":
        o = o.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

    return o, r, d, t, info, training_env_id, mask, num_steps


def run_policy_inference(
    policy: torch.nn.Module,
    observations: Tensor,
    experience: Experience,
    training_env_id_start: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
    """Run the policy to get actions and value estimates.

    Returns:
        Tuple of (actions, selected_action_log_probs, values, lstm_state_to_store)
    """
    with torch.no_grad():
        state = PolicyState()
        lstm_h, lstm_c = experience.get_lstm_state(training_env_id_start)
        if lstm_h is not None:
            state.lstm_h = lstm_h
            state.lstm_c = lstm_c

        actions, selected_action_log_probs, _, value, _ = policy(observations, state)

        if __debug__:
            assert_shape(selected_action_log_probs, ("BT",), "selected_action_log_probs")
            assert_shape(actions, ("BT", 2), "actions")

        lstm_state_to_store = None
        if state.lstm_h is not None and state.lstm_c is not None:
            lstm_state_to_store = {"lstm_h": state.lstm_h.detach(), "lstm_c": state.lstm_c.detach()}

    return actions, selected_action_log_probs, value.flatten(), lstm_state_to_store


def get_lstm_config(policy: Any) -> Tuple[int, int]:
    """Extract LSTM configuration from policy."""
    hidden_size = getattr(policy, "hidden_size", 256)
    num_lstm_layers = 2  # Default value

    # Try to get actual number of LSTM layers from policy
    if hasattr(policy, "components") and "_core_" in policy.components:
        lstm_module = policy.components["_core_"]
        if hasattr(lstm_module, "_net") and hasattr(lstm_module._net, "num_layers"):
            num_lstm_layers = lstm_module._net.num_layers

    return hidden_size, num_lstm_layers
