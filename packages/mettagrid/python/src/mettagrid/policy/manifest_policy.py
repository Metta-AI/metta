from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mettagrid.policy.mpt_artifact import architecture_from_spec, load_mpt
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.file import parse_uri


class ManifestPolicy(MultiAgentPolicy):
    """Load a policy from a checkpoint using an explicit architecture manifest."""

    short_names = ["manifest"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        checkpoint_uri: str,
        architecture_spec: str | None = None,
        manifest_path: str | None = None,
        device: str = "cpu",
        strict: bool = True,
    ):
        super().__init__(policy_env_info, device=device)

        if architecture_spec is None and manifest_path is None:
            raise ValueError("ManifestPolicy requires architecture_spec or manifest_path")
        if architecture_spec is not None and manifest_path is not None:
            raise ValueError("ManifestPolicy accepts only one of architecture_spec or manifest_path")

        if manifest_path is not None:
            architecture_spec = Path(manifest_path).read_text()

        assert architecture_spec is not None
        architecture = architecture_from_spec(architecture_spec)

        artifact = load_mpt(checkpoint_uri)
        policy = architecture.make_policy(policy_env_info)
        policy = policy.to(torch.device(device))

        missing, unexpected = policy.load_state_dict(dict(artifact.state_dict), strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

        if hasattr(policy, "initialize_to_environment"):
            policy.initialize_to_environment(policy_env_info, torch.device(device))

        self._policy = policy
        self._architecture = architecture

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "ManifestPolicy":
        self._policy.eval()
        return self

    def save_policy(
        self,
        destination: str | Path,
        *,
        policy_architecture: Any | None = None,
    ) -> str:
        from mettagrid.policy.mpt_artifact import save_mpt

        architecture = policy_architecture or self._architecture
        if architecture is None:
            raise ValueError("policy_architecture is required to save policy")

        save_mpt(str(destination), architecture=architecture, state_dict=self._policy.state_dict())
        parsed = parse_uri(str(destination), allow_none=False)
        return parsed.canonical
