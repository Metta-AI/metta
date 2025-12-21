from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol


def architecture_from_spec(spec: str) -> Any:
    spec = spec.strip()
    if not spec:
        raise ValueError("architecture_spec cannot be empty")

    class_path = spec.split("(")[0].strip()
    config_class = load_symbol(class_path)
    if not isinstance(config_class, type):
        raise TypeError(f"Loaded symbol {class_path} is not a class")
    if not hasattr(config_class, "from_spec"):
        raise TypeError(f"Class {class_path} does not have a from_spec method")
    return config_class.from_spec(spec)


class CheckpointPolicy(MultiAgentPolicy):
    """Policy implementation that loads weights from a policy_spec directory.

    Expected PolicySpec fields:
      - init_kwargs.architecture_spec: serialized architecture (via to_spec())
      - data_path: safetensors weights file (resolved to local path by PolicySpec loader)
    """

    short_names = ["checkpoint"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        architecture_spec: str,
        device: str = "cpu",
        strict: bool = True,
    ):
        super().__init__(policy_env_info, device=device)
        self._strict = strict
        self._device = torch.device(device)
        self._policy_env_info = policy_env_info
        self._architecture = architecture_from_spec(architecture_spec)
        self._policy = self._architecture.make_policy(policy_env_info).to(self._device)
        self._policy.eval()

    def load_policy_data(self, policy_data_path: str) -> None:
        weights_blob = Path(policy_data_path).read_bytes()
        state_dict = load_safetensors(weights_blob)
        missing, unexpected = self._policy.load_state_dict(dict(state_dict), strict=self._strict)
        if self._strict and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

        if hasattr(self._policy, "initialize_to_environment"):
            self._policy.initialize_to_environment(self._policy_env_info, self._device)
        self._policy.eval()

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "CheckpointPolicy":
        self._policy.eval()
        return self

    @property
    def wrapped_policy(self) -> Any:
        return self._policy
