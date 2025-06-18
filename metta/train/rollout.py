"""Rollout collection for experience gathering."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import torch

from metta.agent.policy_state import PolicyState
from metta.rl.experience import Experience


@dataclass
class RolloutConfig:
    """Configuration for rollout collection."""

    num_steps: int = 128
    device: str = "cuda"
    cpu_offload: bool = False


def rollout(
    agent,
    vecenv,
    experience: Experience,
    config: RolloutConfig,
) -> Dict[str, Any]:
    """Collect rollout experience from environments.

    Args:
        agent: The policy/agent to collect experience with
        vecenv: Vectorized environment
        experience: Experience buffer to store trajectories
        config: Rollout configuration

    Returns:
        Dictionary containing collected statistics
    """
    stats = defaultdict(list)
    experience.reset_for_rollout()

    while not experience.ready_for_training:
        # Receive observations from environments
        obs, rewards, dones, truncs, info, env_id, mask = vecenv.recv()

        # Convert to tensors
        obs_tensor = torch.as_tensor(obs)
        rewards_tensor = torch.as_tensor(rewards)
        dones_tensor = torch.as_tensor(dones)
        truncs_tensor = torch.as_tensor(truncs)

        # Get agent predictions
        with torch.no_grad():
            state = PolicyState()

            # Get LSTM state if using RNN
            lstm_state = experience.get_lstm_state(env_id[0])
            if lstm_state is not None:
                state.lstm_h = lstm_state["lstm_h"]
                state.lstm_c = lstm_state["lstm_c"]

            # Move to device
            obs_device = obs_tensor.to(config.device, non_blocking=True)

            # Get actions from agent
            actions, log_probs, _, values, _ = agent(obs_device, state)

            # Prepare LSTM state for storage
            lstm_state_to_store = None
            if state.lstm_h is not None:
                lstm_state_to_store = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c}

        # Store in experience buffer
        experience.store(
            obs=obs_tensor if config.cpu_offload else obs_device,
            actions=actions,
            logprobs=log_probs,
            rewards=rewards_tensor.to(config.device, non_blocking=True),
            dones=dones_tensor.to(config.device, non_blocking=True),
            truncations=truncs_tensor.to(config.device, non_blocking=True),
            values=values.flatten(),
            env_id=slice(env_id[0], env_id[-1] + 1),
            mask=torch.as_tensor(mask),
            lstm_state=lstm_state_to_store,
        )

        # Collect stats from info dicts
        for i in info:
            for k, v in _unroll_nested_dict(i):
                stats[k].append(v)

        # Send actions to environments
        actions_np = actions.cpu().numpy()
        vecenv.send(actions_np)

    return dict(stats)


def _unroll_nested_dict(d: Dict[str, Any], prefix: str = "") -> list:
    """Unroll nested dictionary into flat key-value pairs."""
    items = []
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_unroll_nested_dict(v, f"{key}/"))
        else:
            items.append((key, v))
    return items
