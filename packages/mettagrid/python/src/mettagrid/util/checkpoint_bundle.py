from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from mettagrid.policy.policy import PolicySpec
from mettagrid.util.file import write_data
from mettagrid.util.uri_resolvers.schemes import checkpoint_filename, parse_uri, resolve_uri


def _join_uri(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child}"


def bundle_dir_from_mpt_uri(mpt_uri: str) -> str:
    if mpt_uri.endswith("policy.mpt"):
        return mpt_uri[: -len("policy.mpt")].rstrip("/")
    return mpt_uri.rsplit("/", 1)[0].rstrip("/")


def submission_zip_uri_from_mpt_uri(mpt_uri: str) -> str:
    return _join_uri(bundle_dir_from_mpt_uri(mpt_uri), "submission.zip")


@dataclass(frozen=True)
class CheckpointBundle:
    """Represents a checkpoint directory containing policy_spec.json and policy.mpt."""

    dir_uri: str
    local_dir: Path | None = None

    @property
    def policy_spec_uri(self) -> str:
        return _join_uri(self.dir_uri, "policy_spec.json")

    @property
    def policy_mpt_uri(self) -> str:
        return _join_uri(self.dir_uri, "policy.mpt")

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
    from mettagrid.policy.mpt_artifact import save_mpt  # Local import to avoid circular dependency
    checkpoint_dir = base_dir / checkpoint_filename(run_name, epoch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mpt_path = checkpoint_dir / "policy.mpt"
    mpt_uri = save_mpt(mpt_path, architecture=architecture, state_dict=state_dict)

    spec = PolicySpec(
        class_path="mettagrid.policy.mpt_policy.MptPolicy",
        init_kwargs={"checkpoint_uri": mpt_uri},
    )
    spec_path = checkpoint_dir / "policy_spec.json"
    spec_path.write_text(spec.model_dump_json())

    return CheckpointBundle(dir_uri=checkpoint_dir.as_uri(), local_dir=checkpoint_dir)


def upload_bundle(bundle: CheckpointBundle, remote_prefix: str) -> CheckpointBundle:
    """Copy a local bundle to a remote prefix (file:// or s3://)."""
    if bundle.local_dir is None:
        raise ValueError("upload_bundle requires a local bundle with a local_dir")

    remote_dir = _join_uri(remote_prefix, bundle.local_dir.name)
    remote_mpt_uri = _join_uri(remote_dir, "policy.mpt")
    write_data(remote_mpt_uri, (bundle.local_dir / "policy.mpt").read_bytes())

    local_spec = PolicySpec.model_validate_json((bundle.local_dir / "policy_spec.json").read_text())
    local_spec.init_kwargs["checkpoint_uri"] = remote_mpt_uri
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
            if parsed.local_path.name != "policy_spec.json":
                raise ValueError("Checkpoint URI must point to a checkpoint directory or policy_spec.json")
            local_dir = parsed.local_path.parent
            dir_uri = local_dir.as_uri()
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
    """
    Resolve a URI that may point to a checkpoint directory or :latest suffix
    into the canonical policy_spec.json URI.
    """
    return resolve_checkpoint_bundle(uri).policy_spec_uri


def resolve_policy_mpt_uri(uri: str) -> str:
    """
    Resolve a URI to the canonical policy.mpt location.

    Accepts:
      - direct .mpt file paths/URIs
      - checkpoint directories
      - :latest on a checkpoint directory
    """
    if uri.endswith(".mpt"):
        return parse_uri(uri).canonical

    return resolve_checkpoint_bundle(uri).policy_mpt_uri
