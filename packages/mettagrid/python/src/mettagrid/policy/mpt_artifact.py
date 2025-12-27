from __future__ import annotations

import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.file import local_copy, parse_uri, write_file
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
    """Load an .mpt checkpoint from a URI.

    Supports file://, s3://, metta://, local paths, and :latest suffix.
    """
    parsed = resolve_uri(uri)
    with local_copy(parsed.canonical) as local_path:
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

    architecture = _maybe_update_architecture_from_state_dict(architecture, state_dict)
    return MptArtifact(architecture=architecture, state_dict=state_dict)


def architecture_from_spec(spec: str) -> PolicyArchitectureProtocol:
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


def _infer_cortex_block_count(state_dict: Mapping[str, torch.Tensor]) -> int | None:
    pattern = re.compile(r"cortex\.func\.stack\.blocks\.(\d+)\.")
    indices: set[int] = set()
    for key in state_dict.keys():
        match = pattern.search(key)
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        return None
    return max(indices) + 1


def _infer_cortex_pattern(state_dict: Mapping[str, torch.Tensor]) -> str | None:
    keys = state_dict.keys()
    if any("cell.net.weight_ih_l0" in key for key in keys):
        return "L"
    if any("cell.nu_log" in key for key in keys):
        return "A"
    return None


def _infer_has_gtd_aux(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any("gtd_aux." in key for key in state_dict.keys())


def _maybe_update_architecture_from_state_dict(architecture: Any, state_dict: Mapping[str, torch.Tensor]) -> Any:
    if getattr(architecture, "components", None):
        return architecture

    updates: dict[str, Any] = {}
    if hasattr(architecture, "core_resnet_pattern"):
        inferred_pattern = _infer_cortex_pattern(state_dict)
        if inferred_pattern and getattr(architecture, "core_resnet_pattern", None) != inferred_pattern:
            updates["core_resnet_pattern"] = inferred_pattern

    if hasattr(architecture, "core_resnet_layers"):
        inferred_layers = _infer_cortex_block_count(state_dict)
        if inferred_layers is not None and getattr(architecture, "core_resnet_layers", None) != inferred_layers:
            updates["core_resnet_layers"] = inferred_layers

    if hasattr(architecture, "include_gtd_aux"):
        if not _infer_has_gtd_aux(state_dict) and getattr(architecture, "include_gtd_aux", True):
            updates["include_gtd_aux"] = False

    if not updates:
        return architecture

    if hasattr(architecture, "model_copy"):
        return architecture.model_copy(update=updates)

    data = architecture.model_dump() if hasattr(architecture, "model_dump") else dict(architecture.__dict__)
    data.update(updates)
    return type(architecture)(**data)
