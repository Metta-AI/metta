"""PufferLib-trained policy shim for CoGames submissions.

This bridges checkpoints produced by PufferLib training (state_dict of
``pufferlib.environments.cogames.torch.Policy``) to the CoGames
`MultiAgentPolicy` interface so they can be used with ``cogames eval`` /
``cogames submit`` without requiring the full PufferLib repo at runtime.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch

import pufferlib.models  # type: ignore[import-untyped]
import pufferlib.pytorch  # type: ignore[import-untyped]
from mettagrid.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class _ShimEnv:
    def __init__(self, policy_env_info: PolicyEnvInterface):
        self.single_observation_space = policy_env_info.observation_space
        self.single_action_space = policy_env_info.action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.env = self
        self.num_agents = policy_env_info.num_agents


class _PufferlibCogsNet(pufferlib.models.Default):
    def __init__(self, env: _ShimEnv, hidden_size: int = 256):
        super().__init__(env, hidden_size=hidden_size)
        self.register_buffer("_inv_scale", torch.tensor(1.0 / 255.0), persistent=False)

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        flattened = observations.view(batch_size, -1).float() * self._inv_scale
        return self.encoder(flattened)


_OBS_PAD_VALUE = 255.0


def _pack_observation(obs: AgentObservation, num_tokens: int, token_dim: int, device: torch.device) -> torch.Tensor:
    buffer = torch.full((num_tokens, token_dim), fill_value=_OBS_PAD_VALUE, device=device, dtype=torch.float32)
    for idx, token in enumerate(obs.tokens):
        if idx >= num_tokens:
            break
        raw = torch.as_tensor(token.raw_token, device=device, dtype=buffer.dtype)
        buffer[idx, : raw.numel()] = raw
    return buffer


class _PufferlibCogsAgent(AgentPolicy):
    def __init__(self, net: _PufferlibCogsNet, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._net = net
        self._device = next(net.parameters()).device
        self._action_names = policy_env_info.action_names
        self._num_tokens, self._token_dim = policy_env_info.observation_space.shape

    def _to_tensor(self, obs: Union[AgentObservation, torch.Tensor, Sequence[Any]]) -> torch.Tensor:
        if isinstance(obs, AgentObservation):
            return _pack_observation(obs, self._num_tokens, self._token_dim, self._device)
        return torch.as_tensor(obs, device=self._device)

    def step(self, obs: Union[AgentObservation, torch.Tensor, Sequence[Any]]) -> Action:
        obs_tensor = self._to_tensor(obs).to(dtype=torch.float32)
        if obs_tensor.ndim == 2:  # Single agent observation
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
        action_idx = max(0, min(int(action.item()), len(self._action_names) - 1))
        return Action(name=self._action_names[action_idx])


class PufferlibCogsPolicy(TrainablePolicy):
    """Loads and runs checkpoints trained with PufferLib's CoGames policy."""

    short_names = ["pufferlib", "pufferlib_cogs"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        hidden_size: int = 256,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(policy_env_info)
        shim_env = _ShimEnv(policy_env_info)
        self._net = _PufferlibCogsNet(shim_env, hidden_size=hidden_size)
        if device is not None:
            self._net = self._net.to(torch.device(device))

    def network(self) -> torch.nn.Module:  # type: ignore[override]
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:  # type: ignore[override]
        return _PufferlibCogsAgent(self._net, self.policy_env_info)

    def is_recurrent(self) -> bool:
        return False

    def load_policy_data(self, policy_data_path: str) -> None:
        state = torch.load(policy_data_path, map_location=next(self._net.parameters()).device)
        self._net.load_state_dict(state)
        self._net = self._net.to(next(self._net.parameters()).device)

    def save_policy_data(self, policy_data_path: str) -> None:
        torch.save(self._net.state_dict(), policy_data_path)
