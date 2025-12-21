from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from mettagrid.policy.checkpoint_policy import CheckpointDir, CheckpointPolicy, WEIGHTS_FILENAME
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.file import write_data
from mettagrid.util.uri_resolvers.schemes import resolve_uri


def _join_uri(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child}"


def write_checkpoint_dir(
    *,
    base_dir: Path,
    run_name: str,
    epoch: int,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> CheckpointDir:
    return CheckpointPolicy.write_checkpoint_dir(
        base_dir=base_dir,
        run_name=run_name,
        epoch=epoch,
        architecture=architecture,
        state_dict=state_dict,
    )


def upload_checkpoint_dir(checkpoint: CheckpointDir, remote_prefix: str) -> CheckpointDir:
    """Copy a local checkpoint directory to a remote prefix (file:// or s3://)."""
    if checkpoint.local_dir is None:
        raise ValueError("upload_checkpoint_dir requires a local checkpoint with a local_dir")

    remote_dir = _join_uri(remote_prefix, checkpoint.local_dir.name)
    remote_weights_uri = _join_uri(remote_dir, WEIGHTS_FILENAME)
    write_data(remote_weights_uri, (checkpoint.local_dir / WEIGHTS_FILENAME).read_bytes())

    local_spec = SubmissionPolicySpec.model_validate_json((checkpoint.local_dir / POLICY_SPEC_FILENAME).read_text())
    local_spec.data_path = WEIGHTS_FILENAME
    write_data(
        _join_uri(remote_dir, POLICY_SPEC_FILENAME),
        local_spec.model_dump_json().encode("utf-8"),
        content_type="application/json",
    )
    return CheckpointDir(dir_uri=remote_dir)


def resolve_checkpoint_dir(uri: str) -> CheckpointDir:
    """Normalize a checkpoint URI (directory or :latest) to a checkpoint directory."""
    parsed = resolve_uri(uri)
    dir_uri = parsed.canonical
    local_dir: Path | None = None

    if parsed.local_path:
        if parsed.local_path.is_file():
            if parsed.local_path.name == POLICY_SPEC_FILENAME:
                local_dir = parsed.local_path.parent
                dir_uri = local_dir.as_uri()
            else:
                raise ValueError("Checkpoint URI must point to a checkpoint directory or policy_spec.json")
        else:
            spec_path = parsed.local_path / POLICY_SPEC_FILENAME
            if not spec_path.exists():
                raise ValueError(f"Checkpoint directory missing policy_spec.json: {parsed.local_path}")
            local_dir = parsed.local_path
            dir_uri = parsed.local_path.as_uri()
    else:
        if dir_uri.endswith(POLICY_SPEC_FILENAME):
            dir_uri = dir_uri[: -len(POLICY_SPEC_FILENAME)].rstrip("/")

    return CheckpointDir(dir_uri=dir_uri, local_dir=local_dir)
