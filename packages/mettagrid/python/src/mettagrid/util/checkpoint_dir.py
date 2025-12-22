from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from mettagrid.policy.checkpoint_policy import WEIGHTS_FILENAME, CheckpointPolicy
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.file import write_data
from mettagrid.util.uri_resolvers.schemes import resolve_uri
def write_checkpoint_dir(
    *,
    base_dir: Path,
    run_name: str,
    epoch: int,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> Path:
    return CheckpointPolicy.write_checkpoint_dir(
        base_dir=base_dir,
        run_name=run_name,
        epoch=epoch,
        architecture=architecture,
        state_dict=state_dict,
    )
def upload_checkpoint_dir(checkpoint_dir: Path, remote_prefix: str) -> str:
    if not checkpoint_dir.is_dir():
        raise ValueError("upload_checkpoint_dir requires a checkpoint directory")

    remote_dir = f"{remote_prefix.rstrip('/')}/{checkpoint_dir.name}"
    remote_weights_uri = f"{remote_dir}/{WEIGHTS_FILENAME}"
    write_data(remote_weights_uri, (checkpoint_dir / WEIGHTS_FILENAME).read_bytes())

    local_spec = SubmissionPolicySpec.model_validate_json((checkpoint_dir / POLICY_SPEC_FILENAME).read_text())
    local_spec.data_path = WEIGHTS_FILENAME
    policy_spec_uri = f"{remote_dir}/{POLICY_SPEC_FILENAME}"
    write_data(policy_spec_uri, local_spec.model_dump_json().encode("utf-8"), content_type="application/json")
    return remote_dir
def resolve_checkpoint_dir(uri: str) -> str:
    parsed = resolve_uri(uri)
    dir_uri = parsed.canonical

    if parsed.local_path:
        if parsed.local_path.is_file():
            if parsed.local_path.name == POLICY_SPEC_FILENAME:
                dir_uri = parsed.local_path.parent.as_uri()
            else:
                raise ValueError("Checkpoint URI must point to a checkpoint directory or policy_spec.json")
        else:
            spec_path = parsed.local_path / POLICY_SPEC_FILENAME
            if not spec_path.exists():
                raise ValueError(f"Checkpoint directory missing policy_spec.json: {parsed.local_path}")
            dir_uri = parsed.local_path.as_uri()
    else:
        if dir_uri.endswith(POLICY_SPEC_FILENAME):
            dir_uri = dir_uri[: -len(POLICY_SPEC_FILENAME)].rstrip("/")

    return dir_uri
