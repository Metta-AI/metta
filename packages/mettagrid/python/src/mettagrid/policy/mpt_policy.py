from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mettagrid.policy.mpt_artifact import load_mpt, save_mpt
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.file import ParsedURI
from mettagrid.util.module import load_symbol

DEFAULT_URI_RESOLVER = "metta.rl.policy_uri_resolver.resolve_uri"


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
        uri_resolver: str | None = DEFAULT_URI_RESOLVER,
        device: str | torch.device = "cpu",
        strict: bool = True,
        display_name: str | None = None,
    ):
        super().__init__(policy_env_info)
        torch_device = torch.device(device) if isinstance(device, str) else device

        resolved_uri = checkpoint_uri
        if uri_resolver and (resolver_func := load_symbol(uri_resolver, strict=False)):
            resolved_uri = resolver_func(checkpoint_uri)  # type: ignore

        self._artifact = load_mpt(resolved_uri)
        self._architecture = self._artifact.architecture

        policy = self._artifact.instantiate(policy_env_info, device=torch_device, strict=strict)
        policy.eval()
        self._policy = policy

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

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
