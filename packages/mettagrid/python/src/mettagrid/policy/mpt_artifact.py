from __future__ import annotations

import copy
import logging
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


def _detect_actor_head_action_count(state_dict: Mapping[str, torch.Tensor]) -> int | None:
    """Return actor_head action count if found, else None."""
    for key, tensor in state_dict.items():
        if "actor_head" in key and "weight" in key:
            return tensor.shape[0]
    return None


def _pad_actor_head_weights(
    state_dict: MutableMapping[str, torch.Tensor],
    checkpoint_actions: int,
    target_actions: int,
) -> None:
    """Pad actor_head weights to match target action count."""
    pad_size = target_actions - checkpoint_actions

    for key in list(state_dict.keys()):
        if "actor_head" not in key:
            continue

        tensor = state_dict[key]
        if "weight" in key:
            pad = torch.zeros(pad_size, tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
            state_dict[key] = torch.cat([tensor, pad], dim=0)
        elif "bias" in key:
            pad = torch.full((pad_size,), float("-inf"), dtype=tensor.dtype, device=tensor.device)
            state_dict[key] = torch.cat([tensor, pad], dim=0)


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
        device: torch.device | str = "cpu",
        *,
        strict: bool = True,
        pad_action_space: bool = False,
    ) -> Any:
        """Instantiate a policy with this state dict."""
        if isinstance(device, str):
            device = torch.device(device)

        # Work on a deep copy so padding does not mutate the artifact across calls.
        state_dict: MutableMapping[str, torch.Tensor] = copy.deepcopy(self.state_dict)

        env_actions = int(policy_env_info.action_space.n)
        checkpoint_actions = _detect_actor_head_action_count(state_dict)

        if checkpoint_actions is not None and checkpoint_actions < env_actions:
            if pad_action_space:
                logging.warning(
                    "Action space mismatch: checkpoint has %d actions, environment expects %d. "
                    "Padding actor_head weights (new actions will have ~0 probability).",
                    checkpoint_actions,
                    env_actions,
                )
                _pad_actor_head_weights(state_dict, checkpoint_actions, env_actions)
            else:
                raise ValueError(
                    f"Action space mismatch: checkpoint has {checkpoint_actions} actions, "
                    f"environment expects {env_actions}. "
                    f"Set pad_action_space=True to pad weights automatically."
                )

        policy = self.architecture.make_policy(policy_env_info)
        policy = policy.to(device)

        missing, unexpected = policy.load_state_dict(dict(state_dict), strict=strict)
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

    return MptArtifact(architecture=architecture, state_dict=state_dict)


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
) -> str:
    """Save an .mpt checkpoint to a URI or local path. Returns the saved URI."""
    parsed = ParsedURI.parse(str(uri))

    if parsed.scheme == "s3":
        with tempfile.NamedTemporaryFile(suffix=".mpt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _save_mpt_file_locally(tmp_path, architecture=architecture, state_dict=state_dict)
            write_file(parsed.canonical, str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        return parsed.canonical
    else:
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
