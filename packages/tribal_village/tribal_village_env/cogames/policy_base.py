"""Reusable base class for PufferLib-backed policies."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch

import pufferlib.models  # type: ignore[import-untyped]
import pufferlib.pytorch  # type: ignore[import-untyped]
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.simulator import Action, AgentObservation, Simulation

ObservationAdapter = Callable[[Union[AgentObservation, np.ndarray, torch.Tensor, Sequence[Any]]], np.ndarray]
ShimFactory = Callable[[Any], Any]


def default_obs_adapter(
    obs: Union[AgentObservation, np.ndarray, torch.Tensor, Sequence[Any]],
    *,
    env_info: Any,
) -> np.ndarray:
    if isinstance(obs, AgentObservation):
        shape = env_info.observation_space.shape
        obs_array = np.zeros(shape, dtype=np.float32)
    else:
        obs_array = np.asarray(obs, dtype=np.float32)
    return obs_array


class DefaultPufferPolicy(MultiAgentPolicy, AgentPolicy):
    """Shared implementation for simple feed-forward policies."""

    def __init__(
        self,
        policy_env_info: Any,
        *,
        hidden_size: int = 256,
        device: Optional[Union[str, torch.device]] = None,
        shim_factory: ShimFactory,
        obs_adapter: Optional[Callable[[Any], np.ndarray]] = None,
    ) -> None:
        MultiAgentPolicy.__init__(self, policy_env_info)
        AgentPolicy.__init__(self, policy_env_info)
        shim_env = shim_factory(policy_env_info)
        if obs_adapter is None:
            obs_adapter = lambda obs: default_obs_adapter(obs, env_info=policy_env_info)  # noqa: E731
        self._obs_adapter = obs_adapter
        self._net = pufferlib.models.Default(shim_env, hidden_size=hidden_size)  # type: ignore[arg-type]
        if device is not None:
            self._net = self._net.to(torch.device(device))

        self._action_names = policy_env_info.action_names
        self._device = next(self._net.parameters()).device
        self._num_actions = len(self._action_names)
        self._policy_env_info = policy_env_info

    def network(self) -> torch.nn.Module:  # type: ignore[override]
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:  # type: ignore[override]
        return self

    def is_recurrent(self) -> bool:
        return False

    def reset(self, simulation: Optional[Simulation] = None) -> None:  # type: ignore[override]
        return None

    def load_policy_data(self, policy_data_path: str) -> None:
        state = torch.load(policy_data_path, map_location=self._device)
        self._net.load_state_dict(state)
        self._net = self._net.to(self._device)

    def save_policy_data(self, policy_data_path: str) -> None:
        torch.save(self._net.state_dict(), policy_data_path)

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        obs_array = self._obs_adapter(obs)
        obs_tensor = torch.as_tensor(obs_array, device=self._device, dtype=torch.float32)
        if obs_tensor.ndim == len(self._policy_env_info.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor * (1.0 / 255.0)

    def step(self, obs: Any) -> Action:  # type: ignore[override]
        obs_tensor = self._obs_to_tensor(obs)
        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            sampled, _, _ = pufferlib.pytorch.sample_logits(logits)
        action_idx = int(sampled.item()) % max(1, self._num_actions)
        return Action(name=self._action_names[action_idx])


__all__ = ["DefaultPufferPolicy", "default_obs_adapter"]
