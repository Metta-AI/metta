from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mettagrid.policy.checkpoint_policy import architecture_from_spec
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.file import local_copy, write_file
from mettagrid.util.uri_resolvers.schemes import parse_uri


class PolicyArchitectureProtocol(Protocol):
    def make_policy(self, policy_env_info: PolicyEnvInterface) -> Any: ...

    def to_spec(self) -> str:
        """Serialize this architecture to a string specification."""
        ...

    @classmethod
    def from_spec(cls, spec: str) -> "PolicyArchitectureProtocol":
        """Deserialize an architecture from a string specification."""
        ...


@dataclass
class MptArtifact:
    architecture: Any
    state_dict: MutableMapping[str, torch.Tensor]

    def instantiate(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        *,
        strict: bool = True,
    ) -> Any:
        torch_device = torch.device(device)

        policy = self.architecture.make_policy(policy_env_info)
        policy = policy.to(torch_device)

        missing, unexpected = policy.load_state_dict(dict(self.state_dict), strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

        if hasattr(policy, "initialize_to_environment"):
            policy.initialize_to_environment(policy_env_info, torch_device)

        return policy


def load_mpt(uri: str) -> MptArtifact:
    """Load an .mpt checkpoint from a local path or s3:// URI."""
    with local_copy(uri) as local_path:
        return _load_local_mpt_file(local_path)


def _load_local_mpt_file(path: Path) -> MptArtifact:
    if not path.exists():
        raise FileNotFoundError(f"MPT file not found: {path}")

    with zipfile.ZipFile(path, mode="r") as archive:
        names = set(archive.namelist())

        if "weights.safetensors" not in names:
            raise ValueError(f"Invalid .mpt file: {path} (missing weights)")

        if "modelarchitecture.txt" in names:
            architecture_blob = archive.read("modelarchitecture.txt").decode("utf-8")
        else:
            raise ValueError(f"Invalid .mpt file: {path} (missing architecture)")

        architecture = architecture_from_spec(architecture_blob)

        weights_blob = archive.read("weights.safetensors")
        state_dict = load_safetensors(weights_blob)
        if not isinstance(state_dict, MutableMapping):
            raise TypeError("Loaded safetensors state_dict is not a mutable mapping")

    return MptArtifact(architecture=architecture, state_dict=state_dict)


def save_mpt(
    uri: str | Path,
    *,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Save an .mpt checkpoint to a URI or local path. Returns the saved URI."""
    parsed = parse_uri(str(uri), allow_none=False)

    if parsed.scheme == "s3":
        with tempfile.NamedTemporaryFile(suffix=".mpt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _save_mpt_file_locally(tmp_path, architecture=architecture, state_dict=state_dict)
            write_file(parsed.canonical, str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        return parsed.canonical

    output_path = parsed.local_path or Path(str(uri)).expanduser().resolve()
    _save_mpt_file_locally(output_path, architecture=architecture, state_dict=state_dict)
    return f"file://{output_path.resolve()}"


def _save_mpt_file_locally(
    path: Path,
    *,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared_state = _prepare_state_dict_for_save(state_dict)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

        try:
            with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                weights_blob = save_safetensors(dict(prepared_state))
                archive.writestr("weights.safetensors", weights_blob)
                archive.writestr("modelarchitecture.txt", architecture.to_spec())

            temp_path.replace(path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise


def _prepare_state_dict_for_save(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Prepare state dict for safetensors: detach, move to CPU, handle shared storage."""
    result: dict[str, torch.Tensor] = {}
    seen_storage: dict[int, str] = {}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State dict entry '{key}' is not a torch.Tensor")

        value = tensor.detach().cpu()
        data_ptr = value.data_ptr()

        if data_ptr in seen_storage:
            value = value.clone()
        else:
            seen_storage[data_ptr] = key

        result[key] = value

    return result


class MptPolicy(MultiAgentPolicy):
    """Load a policy from an .mpt checkpoint file."""

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
        self._device = torch.device(device)
        self._strict = strict
        self._policy_env_info = policy_env_info
        self._policy: Any | None = None
        self._architecture: Any | None = None

        if checkpoint_uri:
            self._load_from_mpt(checkpoint_uri)

    def _load_from_mpt(self, checkpoint_uri: str) -> None:
        artifact = load_mpt(checkpoint_uri)
        self._architecture = artifact.architecture
        self._policy = artifact.instantiate(self._policy_env_info, device=str(self._device), strict=self._strict)
        self._policy.eval()

    def load_policy_data(self, policy_data_path: str) -> None:
        self._load_from_mpt(policy_data_path)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if self._policy is None:
            raise RuntimeError("MptPolicy has not been initialized with checkpoint data")
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "MptPolicy":
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
        if self._policy is None:
            raise ValueError("Policy has not been loaded; cannot save")
        architecture = policy_architecture or self._architecture
        if architecture is None:
            raise ValueError("policy_architecture is required to save policy")

        return save_mpt(str(destination), architecture=architecture, state_dict=self._policy.state_dict())
