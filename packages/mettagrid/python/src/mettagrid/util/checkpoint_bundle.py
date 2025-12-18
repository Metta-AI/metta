from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save as save_safetensors

from mettagrid.policy.policy import PolicySpec
from mettagrid.util.file import write_data
from mettagrid.util.uri_resolvers.schemes import checkpoint_filename, resolve_uri


def _join_uri(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child}"


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

@dataclass(frozen=True)
class CheckpointBundle:
    """Represents a checkpoint directory containing policy_spec.json and its artifacts."""

    dir_uri: str
    local_dir: Path | None = None

    @property
    def policy_spec_uri(self) -> str:
        return _join_uri(self.dir_uri, "policy_spec.json")

    @property
    def policy_data_uri(self) -> str:
        return _join_uri(self.dir_uri, "weights.safetensors")

    @property
    def submission_zip_uri(self) -> str:
        return _join_uri(self.dir_uri, "submission.zip")


def create_local_bundle(
    *,
    base_dir: Path,
    run_name: str,
    epoch: int,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> CheckpointBundle:
    """Write a checkpoint bundle to disk and return its metadata."""
    checkpoint_dir = base_dir / checkpoint_filename(run_name, epoch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    weights_path = checkpoint_dir / "weights.safetensors"
    prepared_state = _prepare_state_dict_for_save(state_dict)
    weights_blob = save_safetensors(dict(prepared_state))
    weights_path.write_bytes(weights_blob)

    spec = PolicySpec(
        class_path="mettagrid.policy.checkpoint_policy.CheckpointPolicy",
        data_path="weights.safetensors",
        init_kwargs={"architecture_spec": architecture.to_spec()},
    )
    spec_path = checkpoint_dir / "policy_spec.json"
    spec_path.write_text(spec.model_dump_json())

    return CheckpointBundle(dir_uri=checkpoint_dir.as_uri(), local_dir=checkpoint_dir)


def upload_bundle(bundle: CheckpointBundle, remote_prefix: str) -> CheckpointBundle:
    """Copy a local bundle to a remote prefix (file:// or s3://)."""
    if bundle.local_dir is None:
        raise ValueError("upload_bundle requires a local bundle with a local_dir")

    remote_dir = _join_uri(remote_prefix, bundle.local_dir.name)
    remote_weights_uri = _join_uri(remote_dir, "weights.safetensors")
    write_data(remote_weights_uri, (bundle.local_dir / "weights.safetensors").read_bytes())

    local_spec = PolicySpec.model_validate_json((bundle.local_dir / "policy_spec.json").read_text())
    local_spec.data_path = "weights.safetensors"
    write_data(
        _join_uri(remote_dir, "policy_spec.json"),
        local_spec.model_dump_json().encode("utf-8"),
        content_type="application/json",
    )
    return CheckpointBundle(dir_uri=remote_dir)


def resolve_checkpoint_bundle(uri: str) -> CheckpointBundle:
    """Normalize a checkpoint URI (directory or :latest) to a bundle."""
    parsed = resolve_uri(uri)
    dir_uri = parsed.canonical
    local_dir: Path | None = None

    if parsed.local_path:
        if parsed.local_path.is_file():
            if parsed.local_path.name == "policy_spec.json":
                local_dir = parsed.local_path.parent
                dir_uri = local_dir.as_uri()
            else:
                raise ValueError("Checkpoint URI must point to a checkpoint directory or policy_spec.json")
        else:
            spec_path = parsed.local_path / "policy_spec.json"
            if not spec_path.exists():
                raise ValueError(f"Checkpoint directory missing policy_spec.json: {parsed.local_path}")
            local_dir = parsed.local_path
            dir_uri = parsed.local_path.as_uri()
    else:
        if dir_uri.endswith("policy_spec.json"):
            dir_uri = dir_uri[: -len("policy_spec.json")].rstrip("/")

    return CheckpointBundle(dir_uri=dir_uri, local_dir=local_dir)


def resolve_policy_spec_uri(uri: str) -> str:
    """Resolve arbitrary checkpoint URI to policy_spec.json URI."""
    return resolve_checkpoint_bundle(uri).policy_spec_uri
