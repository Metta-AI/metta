"""PufferLib-trained policy shim for CoGames submissions.

This bridges checkpoints produced by PufferLib training (state_dict of
``pufferlib.environments.cogames.torch.Policy``) to the CoGames
`MultiAgentPolicy` interface so they can be used with ``cogames eval`` /
``cogames submit`` without requiring the full PufferLib repo at runtime.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Sequence, Union

import pufferlib.models  # type: ignore[import-untyped]
import pufferlib.pytorch  # type: ignore[import-untyped]
import torch
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation


_LSTM_KEYS = ("lstm.", "cell.")


class _PufferlibCogsStatefulImpl(StatefulPolicyImpl[dict[str, torch.Tensor | None]]):
    def __init__(
        self,
        net: torch.nn.Module,
        policy_env_info: PolicyEnvInterface,
        device: torch.device,
        *,
        is_recurrent: bool,
    ) -> None:
        self._net = net
        self._policy_env_info = policy_env_info
        self._action_names = policy_env_info.action_names
        self._num_tokens, self._token_dim = policy_env_info.observation_space.shape
        self._device = device
        self._is_recurrent = is_recurrent

    def reset(self) -> None:
        return None

    def initial_agent_state(self) -> dict[str, torch.Tensor | None]:
        if not self._is_recurrent:
            return {}
        return {"lstm_h": None, "lstm_c": None}

    def step_with_state(
        self,
        obs: AgentObservation,
        state: dict[str, torch.Tensor | None],
    ) -> tuple[Action, dict[str, torch.Tensor | None]]:
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

        obs_tensor = obs_tensor * (1.0 / 255.0)
        obs_tensor = obs_tensor.unsqueeze(0)

        state_dict: dict[str, torch.Tensor | None] | None
        if self._is_recurrent:
            state_dict = {
                "lstm_h": state.get("lstm_h"),
                "lstm_c": state.get("lstm_c"),
            }
        else:
            state_dict = None

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor, state_dict)  # type: ignore[arg-type]
            sampled, _, _ = pufferlib.pytorch.sample_logits(logits)
        action_idx = max(0, min(int(sampled.item()), len(self._action_names) - 1))
        action = Action(name=self._action_names[action_idx])

        if state_dict is None:
            return action, {}
        next_h = state_dict.get("lstm_h")
        next_c = state_dict.get("lstm_c")
        return action, {
            "lstm_h": next_h.detach() if next_h is not None else None,
            "lstm_c": next_c.detach() if next_c is not None else None,
        }


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
        self._hidden_size = hidden_size
        self._device = torch.device(device)
        self._shim_env = SimpleNamespace(
            single_observation_space=policy_env_info.observation_space,
            single_action_space=policy_env_info.action_space,
            observation_space=policy_env_info.observation_space,
            action_space=policy_env_info.action_space,
            num_agents=policy_env_info.num_agents,
        )
        self._shim_env.env = self._shim_env
        self._net = pufferlib.models.Default(self._shim_env, hidden_size=hidden_size)  # type: ignore[arg-type]
        self._net = self._net.to(self._device)
        self._action_names = policy_env_info.action_names
        self._num_tokens, self._token_dim = policy_env_info.observation_space.shape
        self._is_recurrent = False
        self._stateful_impl = _PufferlibCogsStatefulImpl(
            self._net,
            policy_env_info,
            self._device,
            is_recurrent=self._is_recurrent,
        )
        self._agent_policies: dict[int, StatefulAgentPolicy[dict[str, torch.Tensor | None]]] = {}
        self._fallback_agent_policy = StatefulAgentPolicy(self._stateful_impl, policy_env_info)

    def network(self) -> torch.nn.Module:  # type: ignore[override]
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:  # type: ignore[override]
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(self._stateful_impl, self._policy_env_info, agent_id)
        return self._agent_policies[agent_id]

    def is_recurrent(self) -> bool:
        return self._is_recurrent

    def reset(self, simulation: Optional[Simulation] = None) -> None:  # type: ignore[override]
        self._fallback_agent_policy.reset(simulation)
        for policy in self._agent_policies.values():
            policy.reset(simulation)

    def load_policy_data(self, policy_data_path: str) -> None:
        state = torch.load(policy_data_path, map_location=self._device)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        uses_rnn = any(key.startswith(_LSTM_KEYS) for key in state)
        base_net = pufferlib.models.Default(self._shim_env, hidden_size=self._hidden_size)  # type: ignore[arg-type]
        if uses_rnn:
            net: torch.nn.Module = pufferlib.models.LSTMWrapper(
                self._shim_env,
                base_net,
                input_size=base_net.hidden_size,
                hidden_size=base_net.hidden_size,
            )
        else:
            net = base_net
        net.load_state_dict(state)
        net = net.to(self._device)
        self._net = net
        self._is_recurrent = uses_rnn
        self._stateful_impl = _PufferlibCogsStatefulImpl(
            self._net,
            self._policy_env_info,
            self._device,
            is_recurrent=self._is_recurrent,
        )
        self._agent_policies.clear()
        self._fallback_agent_policy = StatefulAgentPolicy(self._stateful_impl, self._policy_env_info)

    def save_policy_data(self, policy_data_path: str) -> None:
        torch.save(self._net.state_dict(), policy_data_path)

    def step(self, obs: Union[AgentObservation, torch.Tensor, Sequence[Any]]) -> Action:  # type: ignore[override]
        if isinstance(obs, AgentObservation):
            return self._fallback_agent_policy.step(obs)
        obs_tensor = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor, None)
            sampled, _, _ = pufferlib.pytorch.sample_logits(logits)
        action_idx = max(0, min(int(sampled.item()), len(self._action_names) - 1))
        return Action(name=self._action_names[action_idx])
