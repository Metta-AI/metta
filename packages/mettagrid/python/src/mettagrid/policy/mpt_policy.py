from __future__ import annotations

from pathlib import Path
from typing import Any

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
        device: str = "cpu",
        strict: bool = True,
    ):
        super().__init__(policy_env_info, device=device)

        artifact = load_mpt(checkpoint_uri)
        self._architecture = artifact.architecture

        self._policy = artifact.instantiate(policy_env_info, device=device, strict=strict)
        self._policy.eval()

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "MptPolicy":
        """Ensure wrapped policy enters eval mode for rollout/play compatibility."""
        self._policy.eval()
        return self

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
