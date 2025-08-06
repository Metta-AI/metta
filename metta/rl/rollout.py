"""Rollout phase functions for Metta training."""

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.common.profiling.stopwatch import Stopwatch
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


PufferlibVecEnv = Any


def get_observation(
    vecenv: PufferlibVecEnv,
    device: torch.device,
    timer: Stopwatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list, slice, Tensor, int]:
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
    vecenv: PufferlibVecEnv,
    actions: Tensor,
    dtype_actions: np.dtype,
    timer: Stopwatch,
) -> None:
    """Send actions back to the vectorized environment."""
    with timer("_rollout.env"):
        vecenv.send(actions.cpu().numpy().astype(dtype_actions))


def run_policy_inference(
    policy: PolicyAgent,
    observations: Tensor,
    experience: Experience,
    training_env_id_start: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor] | None]:
    """Run policy inference to get actions and values."""
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

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    return actions, selected_action_log_probs, value.flatten(), lstm_state_to_store


def get_lstm_config(policy: PolicyAgent) -> tuple[int, int]:
    """Extract LSTM configuration from policy."""
    hidden_size = getattr(policy, "hidden_size", 256)
    num_lstm_layers = 2  # Default value

    # Try to get actual number of LSTM layers from policy
    if hasattr(policy, "components") and "_core_" in policy.components:
        lstm_module = policy.components["_core_"]
        if hasattr(lstm_module, "_net") and hasattr(lstm_module._net, "num_layers"):
            num_lstm_layers = lstm_module._net.num_layers

    return hidden_size, num_lstm_layers
