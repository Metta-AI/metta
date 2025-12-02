from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import numpy as np
import torch
from tensordict import TensorDict

from metta.agent.policy import Policy
from metta.rl.checkpoint_manager import CheckpointManager
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class NPCPolicyLoadError(RuntimeError):
    """Raised when an NPC policy cannot be created from configuration."""


def _load_symbol(path: str) -> Any:
    """Import a symbol from a string path."""

    if ":" in path:
        module_name, symbol_name = path.split(":", 1)
    else:
        module_name, _, symbol_name = path.rpartition(".")
    if not module_name:
        raise ImportError(f"Cannot import '{path}': missing module path")
    module = import_module(module_name)
    return getattr(module, symbol_name)


@dataclass
class _ScriptedEnvController:
    policy: Any
    agent_policies: list[Any]


class ScriptedPolicyAdapter(Policy):
    """Wrap a Cogames-style scripted policy so it can be used as an NPC Policy."""

    def __init__(
        self,
        class_path: str,
        policy_kwargs: Optional[dict[str, Any]],
        policy_env_info: PolicyEnvInterface,
        vector_env: Any | None,
    ) -> None:
        super().__init__(policy_env_info)
        self._policy_cls = _load_symbol(class_path)
        self._policy_kwargs = dict(policy_kwargs or {})
        self._policy_env_info = policy_env_info
        self._vector_env = vector_env
        self._device = torch.device("cpu")
        self._controllers: list[_ScriptedEnvController] = []
        self._agents_per_env: Optional[int] = None

    def initialize_to_environment(self, policy_env_info: PolicyEnvInterface, device: torch.device):
        self._device = torch.device(device)
        self._policy_env_info = policy_env_info

        self._controllers.clear()
        vecenv = self._vector_env
        if vecenv is None:
            # Create a minimal stub environment compatible with scripted policies.
            num_agents = policy_env_info.num_agents
            env_stub = SimpleNamespace(num_agents=num_agents)
            envs: Iterable[Any] = [SimpleNamespace(_env=env_stub, num_agents=num_agents)]
        else:
            envs = getattr(vecenv, "envs", [])
        for env in envs:
            underlying_env = getattr(env, "_env", env)
            scripted_policy = self._policy_cls(underlying_env, **self._policy_kwargs)
            agent_policies = [scripted_policy.agent_policy(i) for i in range(underlying_env.num_agents)]
            self._controllers.append(_ScriptedEnvController(scripted_policy, agent_policies))

        if not self._controllers:
            raise RuntimeError("ScriptedPolicyAdapter requires at least one environment instance")

        if vecenv is None:
            self._agents_per_env = policy_env_info.num_agents
        else:
            first_env = getattr(vecenv, "driver_env", None)
            if first_env is None:
                raise RuntimeError("Vector environment is missing driver_env attribute")
            self._agents_per_env = first_env.num_agents

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        if self._agents_per_env is None:
            raise RuntimeError("ScriptedPolicyAdapter must be initialized before use")

        env_obs = td.get("env_obs")
        if env_obs is None:
            raise KeyError("env_obs tensor missing from rollout TensorDict")

        env_obs = env_obs.to(device="cpu")
        batch = env_obs.shape[0]
        actions = torch.empty(batch, dtype=torch.int64, device=self._device)

        dones = td.get("dones")
        dones_view = dones.reshape(-1) if isinstance(dones, torch.Tensor) else None

        for row in range(batch):
            env_idx = row // self._agents_per_env
            agent_idx = row % self._agents_per_env
            controller = self._controllers[env_idx]
            agent_policy = controller.agent_policies[agent_idx]

            if dones_view is not None and dones_view[row] > 0.5:
                agent_policy.reset()

            obs_np = env_obs[row].detach().contiguous().cpu().numpy()
            if obs_np.ndim == 1:
                obs_np = obs_np.reshape(-1, 3)
            action_id = agent_policy.step(np.asarray(obs_np, dtype=np.uint8))
            actions[row] = int(action_id)

        td.set("actions", actions.to(device=self._device))
        zeros = torch.zeros(batch, dtype=torch.float32, device=self._device)
        td.set("act_log_prob", zeros.clone())
        td.set("entropy", zeros.clone())
        td.set("values", zeros.clone())
        td.set("full_log_probs", zeros.reshape(batch, 1))
        return td

    def reset_memory(self) -> None:
        for controller in self._controllers:
            for agent_policy in controller.agent_policies:
                agent_policy.reset()

    @property
    def device(self) -> torch.device:
        return self._device


def create_scripted_policy_adapter(
    class_path: str,
    policy_kwargs: dict[str, Any],
    policy_env_info: PolicyEnvInterface,
    vector_env: Any | None,
) -> ScriptedPolicyAdapter:
    return ScriptedPolicyAdapter(
        class_path=class_path,
        policy_kwargs=policy_kwargs,
        policy_env_info=policy_env_info,
        vector_env=vector_env,
    )


def load_npc_policy(
    *,
    npc_policy_uri: Optional[str],
    npc_policy_class: Optional[str],
    npc_policy_kwargs: dict[str, Any],
    policy_env_info: PolicyEnvInterface,
    device: torch.device,
    vector_env: Any | None,
) -> tuple[Policy, str]:
    """Instantiate an NPC policy from configuration."""

    if npc_policy_uri:
        policy: Policy = CheckpointManager.load_from_uri(npc_policy_uri, policy_env_info, device)
        return policy, npc_policy_uri

    if npc_policy_class:
        adapter = create_scripted_policy_adapter(
            class_path=npc_policy_class,
            policy_kwargs=npc_policy_kwargs,
            policy_env_info=policy_env_info,
            vector_env=vector_env,
        )
        policy = adapter
        return policy, npc_policy_class

    raise NPCPolicyLoadError("NPC policy configuration must specify either a URI or a class path")
