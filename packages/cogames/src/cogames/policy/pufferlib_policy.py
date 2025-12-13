"""PufferLib-trained policy shim for CoGames submissions.

This bridges checkpoints produced by PufferLib training (state_dict of
``pufferlib.environments.cogames.torch.Policy``) to the CoGames
`MultiAgentPolicy` interface so they can be used with ``cogames eval`` /
``cogames submit`` without requiring the full PufferLib repo at runtime.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Sequence, Union

import torch

import pufferlib.models  # type: ignore[import-untyped]
import pufferlib.pytorch  # type: ignore[import-untyped]
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation


class PufferlibCogsPolicy(MultiAgentPolicy, AgentPolicy):
    """Loads and runs checkpoints trained with PufferLib's CoGames policy.

    This policy serves as both the MultiAgentPolicy factory and AgentPolicy
    implementation, returning itself from agent_policy().
    """

    short_names = ["pufferlib_cogs"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        hidden_size: int = 256,
        device: str = "cpu",
    ):
        MultiAgentPolicy.__init__(self, policy_env_info, device=device)
        AgentPolicy.__init__(self, policy_env_info)
        shim_env = SimpleNamespace(
            single_observation_space=policy_env_info.observation_space,
            single_action_space=policy_env_info.action_space,
            observation_space=policy_env_info.observation_space,
            action_space=policy_env_info.action_space,
            num_agents=policy_env_info.num_agents,
        )
        shim_env.env = shim_env
        self._net = pufferlib.models.Default(shim_env, hidden_size=hidden_size)  # type: ignore[arg-type]
        self._net = self._net.to(torch.device(device))
        self._action_names = policy_env_info.action_names
        self._num_tokens, self._token_dim = policy_env_info.observation_space.shape
        self._device = next(self._net.parameters()).device

    def network(self) -> torch.nn.Module:  # type: ignore[override]
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:  # type: ignore[override]
        return self

    def is_recurrent(self) -> bool:
        return False

    def reset(self, simulation: Optional[Simulation] = None) -> None:  # type: ignore[override]
        # No internal state to reset; signature satisfies AgentPolicy and MultiAgentPolicy
        return None

    def load_policy_data(self, policy_data_path: str) -> None:
        state = torch.load(policy_data_path, map_location=next(self._net.parameters()).device)
        self._net.load_state_dict(state)
        self._net = self._net.to(next(self._net.parameters()).device)

    def save_policy_data(self, policy_data_path: str) -> None:
        torch.save(self._net.state_dict(), policy_data_path)

    def step(self, obs: Union[AgentObservation, torch.Tensor, Sequence[Any]]) -> Action:  # type: ignore[override]
        if isinstance(obs, AgentObservation):
            obs_tensor = torch.full(
                (self._num_tokens, self._token_dim),
                fill_value=255.0,
                device=self._device,
                dtype=torch.float32,
            )
            for idx, token in enumerate(obs.tokens):
                if idx >= self._num_tokens:
                    break
                raw = torch.as_tensor(token.raw_token, device=self._device, dtype=obs_tensor.dtype)
                obs_tensor[idx, : raw.numel()] = raw
        else:
            obs_tensor = torch.as_tensor(obs, device=self._device, dtype=torch.float32)

        obs_tensor = obs_tensor * (1.0 / 255.0)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            sampled, _, _ = pufferlib.pytorch.sample_logits(logits)
        action_idx = max(0, min(int(sampled.item()), len(self._action_names) - 1))
        return Action(name=self._action_names[action_idx])
