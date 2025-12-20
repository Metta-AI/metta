from __future__ import annotations

from pathlib import Path
from typing import Any

from metta.rl.mpt_artifact import load_mpt, save_mpt
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import parse_uri


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
        checkpoint_uri: str | None = None,
        device: str = "cpu",
        strict: bool = True,
    ):
        super().__init__(policy_env_info, device=device)

        self._policy = None
        self._architecture = None
        self._strict = strict
        self._device = device

        if checkpoint_uri:
            self._load_from_checkpoint(checkpoint_uri, device=device)

    def _load_from_checkpoint(self, checkpoint_uri: str, *, device: str) -> None:
        artifact = load_mpt(checkpoint_uri)
        self._architecture = artifact.architecture
        self._policy = artifact.instantiate(self._policy_env_info, device=device, strict=self._strict)
        self._policy.eval()

    def load_policy_data(self, policy_data_path: str) -> None:
        self._load_from_checkpoint(policy_data_path, device=self._device)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if self._policy is None:
            raise RuntimeError("MptPolicy has not been initialized with checkpoint data")
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "MptPolicy":
        """Ensure wrapped policy enters eval mode for rollout/play compatibility."""
        if self._policy is not None:
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
        if self._policy is None:
            raise ValueError("Policy has not been loaded; cannot save")

        save_mpt(str(destination), architecture=architecture, state_dict=self._policy.state_dict())

        parsed = parse_uri(str(destination), allow_none=False)
        return parsed.canonical
