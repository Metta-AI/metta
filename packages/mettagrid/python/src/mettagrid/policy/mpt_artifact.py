from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.file import ParsedURI, local_copy, write_file
from mettagrid.util.module import load_symbol
from mettagrid.util.uri_resolvers.schemes import resolve_uri


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
    env_meta: dict[str, Any] | None = None

    def instantiate(
        self,
        policy_env_info: PolicyEnvInterface,
        device: torch.device | str = "cpu",
        *,
        strict: bool = True,
    ) -> Any:
        if isinstance(device, str):
            device = torch.device(device)

        expected_action_names = policy_env_info.action_names

        # Hard check: if the artifact carries action/obs metadata, enforce an exact match.
        if self.env_meta:
            saved_names = self.env_meta.get("action_names")
            if saved_names is not None and saved_names != expected_action_names:
                raise ValueError(
                    "Action space mismatch between checkpoint and environment. "
                    f"Checkpoint actions={saved_names}, env actions={expected_action_names}"
                )
            saved_obs_shape = tuple(self.env_meta.get("observation_space_shape", []))
            if saved_obs_shape and saved_obs_shape != tuple(policy_env_info.observation_space.shape):
                raise ValueError(
                    "Observation space mismatch between checkpoint and environment. "
                    f"Checkpoint obs_shape={saved_obs_shape}, "
                    f"env obs_shape={tuple(policy_env_info.observation_space.shape)}"
                )

        expected_num_actions = len(expected_action_names)
        saved_active_indices = [
            tensor
            for key, tensor in self.state_dict.items()
            if key.endswith("action_embedding.active_indices")
            and isinstance(tensor, torch.Tensor)
            and tensor.dim() == 1
        ]
        for tensor in saved_active_indices:
            if tensor.numel() != expected_num_actions:
                raise ValueError(
                    "Action space mismatch: checkpoint was saved with "
                    f"{tensor.numel()} actions but environment reports {expected_num_actions}. "
                    "Ensure the checkpoint is used with the same action set/order."
                )

        policy = self.architecture.make_policy(policy_env_info)
        policy = policy.to(device)

        load_state = dict(self.state_dict)

        # Replace any env-dependent buffers that have mismatched shapes with the current buffer
        # values so strict loading succeeds without trying to copy incompatible tensors.
        buffer_map = dict(policy.named_buffers())
        for key, buf in buffer_map.items():
            if key in load_state and hasattr(load_state[key], "shape"):
                if tuple(load_state[key].shape) != tuple(buf.shape):
                    load_state[key] = buf.detach().clone()

        missing, unexpected = policy.load_state_dict(load_state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

        if hasattr(policy, "initialize_to_environment"):
            policy.initialize_to_environment(policy_env_info, device)

        return policy


def load_mpt(uri: str) -> MptArtifact:
    """Load an .mpt checkpoint from a URI.

    Supports file://, s3://, metta://, local paths, and :latest suffix.
    """
    resolved_uri = resolve_uri(uri)
    with local_copy(resolved_uri) as local_path:
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
        architecture = _architecture_from_spec(architecture_blob)

        weights_blob = archive.read("weights.safetensors")
        state_dict = load_safetensors(weights_blob)
        if not isinstance(state_dict, MutableMapping):
            raise TypeError("Loaded safetensors state_dict is not a mutable mapping")

        env_meta = None
        if "envmeta.json" in names:
            try:
                env_meta = json.loads(archive.read("envmeta.json").decode("utf-8"))
            except Exception:
                env_meta = None

    return MptArtifact(architecture=architecture, state_dict=state_dict, env_meta=env_meta)


def _architecture_from_spec(spec: str) -> PolicyArchitectureProtocol:
    """Deserialize an architecture from a string specification."""
    spec = spec.strip()
    if not spec:
        raise ValueError("Policy architecture specification cannot be empty")

    # Extract class path (everything before '(' if present)
    class_path = spec.split("(")[0].strip()

    config_class = load_symbol(class_path)
    if not isinstance(config_class, type):
        raise TypeError(f"Loaded symbol {class_path} is not a class")

    if not hasattr(config_class, "from_spec"):
        raise TypeError(f"Class {class_path} does not have a from_spec method")

    return config_class.from_spec(spec)


def save_mpt(
    uri: str | Path,
    *,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
    policy_env_info: PolicyEnvInterface | None = None,
) -> str:
    """Save an .mpt checkpoint to a URI or local path. Returns the saved URI."""
    parsed = ParsedURI.parse(str(uri))

    env_meta = None
    if policy_env_info is not None:
        env_meta = {
            "action_names": list(policy_env_info.action_names),
            "observation_space_shape": tuple(policy_env_info.observation_space.shape),
            "num_agents": policy_env_info.num_agents,
            "obs_width": policy_env_info.obs_width,
            "obs_height": policy_env_info.obs_height,
        }

    if parsed.scheme == "s3":
        with tempfile.NamedTemporaryFile(suffix=".mpt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _save_mpt_file_locally(tmp_path, architecture=architecture, state_dict=state_dict, env_meta=env_meta)
            write_file(parsed.canonical, str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        return parsed.canonical
    else:
        output_path = parsed.local_path or Path(str(uri)).expanduser().resolve()
        _save_mpt_file_locally(output_path, architecture=architecture, state_dict=state_dict, env_meta=env_meta)
        return f"file://{output_path.resolve()}"


def _save_mpt_file_locally(
    path: Path,
    *,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
    env_meta: dict[str, Any] | None = None,
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
                if env_meta is not None:
                    archive.writestr("envmeta.json", json.dumps(env_meta))

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
