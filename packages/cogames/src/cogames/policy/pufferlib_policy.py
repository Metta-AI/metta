"""PufferLib-trained policy shim for CoGames submissions.

This bridges checkpoints produced by PufferLib training (state_dict of
``pufferlib.environments.cogames.torch.Policy``) to the CoGames
`MultiAgentPolicy` interface so they can be used with ``cogames eval`` /
``cogames submit`` without requiring the full PufferLib repo at runtime.
"""

from __future__ import annotations

import torch

import pufferlib.models
import pufferlib.pytorch
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class _ShimEnv:
    """Minimal object that provides the fields PufferLib policies expect."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        self.single_observation_space = policy_env_info.observation_space
        self.single_action_space = policy_env_info.action_space
        # Some PufferLib helpers look for these aliases
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.env = self  # For policies that access env.env.observation_space
        self.num_agents = policy_env_info.num_agents


class _PufferlibCogsNet(pufferlib.models.Default):
    """Matches the simple PufferLib CoGames policy architecture."""

    def __init__(self, env: _ShimEnv, hidden_size: int = 256):
        super().__init__(env, hidden_size=hidden_size)
        self.register_buffer("_inv_scale", torch.tensor(1.0 / 255.0), persistent=False)

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        flattened = observations.view(batch_size, -1).float() * self._inv_scale
        return self.encoder(flattened)


class _PufferlibCogsAgent(AgentPolicy):
    """Per-agent wrapper that samples actions from the shared network."""

    def __init__(self, net: _PufferlibCogsNet, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._net = net
        self._device = next(net.parameters()).device

    def step(self, obs):
        obs_tensor = torch.as_tensor(obs, device=self._device)
        if obs_tensor.ndim == 2:  # Single agent observation
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
        return dtype_actions.type(int(action.item()))


class PufferlibCogsPolicy(TrainablePolicy):
    """Loads and runs checkpoints trained with PufferLib's CoGames policy."""

    short_names = ["pufferlib", "pufferlib_cogs"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        hidden_size: int = 256,
        device: str | torch.device | None = None,
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
