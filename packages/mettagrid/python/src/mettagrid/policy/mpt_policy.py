from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mettagrid.policy.mpt_artifact import load_mpt, save_mpt
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.file import ParsedURI


class MptPolicy(MultiAgentPolicy):
    """Load a policy from an .mpt checkpoint file.

    The .mpt format stores weights and architecture configuration. This allows
    loading trained policies without a build dependency on the training code.
    """

    short_names = ["mpt"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        checkpoint_uri: str,
        device: str | torch.device = "cpu",
        strict: bool = True,
        display_name: str | None = None,
    ):
        super().__init__(policy_env_info)
        torch_device = torch.device(device) if isinstance(device, str) else device

        self._checkpoint_uri = checkpoint_uri
        self._artifact = load_mpt(checkpoint_uri)
        self._architecture = self._artifact.architecture

        policy = self._artifact.instantiate(policy_env_info, device=torch_device, strict=strict)
        policy.eval()
        self._policy = policy
        self._display_name = display_name or checkpoint_uri

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def architecture(self) -> Any:
        return self._architecture

    @property
    def state_dict_copy(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self._policy.state_dict().items()}

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def __call__(self, *args, **kwargs):
        return self._policy(*args, **kwargs)

    def save_policy(
        self,
        destination: str | Path,
        *,
        policy_architecture: Any | None = None,
    ) -> str:
        """Save the wrapped policy to a URI or local path."""
        architecture = policy_architecture or self._architecture
        if architecture is None:
            raise ValueError("policy_architecture is required to save policy")

        save_mpt(str(destination), architecture=architecture, state_dict=self._policy.state_dict())

        parsed = ParsedURI.parse(str(destination))
        return parsed.canonical

    def __getattr__(self, name: str):
        return getattr(self._policy, name)
