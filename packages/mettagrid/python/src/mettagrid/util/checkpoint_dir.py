from __future__ import annotations

import atexit
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from safetensors.torch import save as save_safetensors

from mettagrid.policy.policy import PolicySpec, architecture_spec_from_value
from mettagrid.policy.prepare_policy_spec import DEFAULT_POLICY_CACHE_DIR, load_policy_spec_from_local_dir
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.file import read, write_data
from mettagrid.util.uri_resolvers.schemes import checkpoint_filename, resolve_uri

WEIGHTS_FILENAME = "weights.safetensors"

_registered_cleanup_dirs: set[Path] = set()


def _join_uri(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child}"


@dataclass(frozen=True)
class CheckpointDir:
    dir_uri: str
    local_dir: Path | None = None

    @property
    def policy_spec_uri(self) -> str:
        return _join_uri(self.dir_uri, POLICY_SPEC_FILENAME)

    @property
    def weights_uri(self) -> str:
        return _join_uri(self.dir_uri, WEIGHTS_FILENAME)

    @property
    def submission_zip_uri(self) -> str:
        return _join_uri(self.dir_uri, "submission.zip")


@dataclass(frozen=True)
class CheckpointBundle:
    dir_uri: str
    local_dir: Path
    policy_spec: PolicySpec
    submission_spec: SubmissionPolicySpec
    weights_path: Path | None

    @property
    def policy_spec_path(self) -> Path:
        return self.local_dir / POLICY_SPEC_FILENAME


def prepare_state_dict_for_save(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def write_checkpoint_dir(
    *,
    base_dir: Path,
    run_name: str,
    epoch: int,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> CheckpointDir:
    architecture_spec = architecture_spec_from_value(architecture)
    checkpoint_dir = (base_dir / checkpoint_filename(run_name, epoch)).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    prepared_state = prepare_state_dict_for_save(state_dict)
    weights_path.write_bytes(save_safetensors(dict(prepared_state)))
    write_policy_spec(checkpoint_dir, architecture_spec)

    return CheckpointDir(dir_uri=checkpoint_dir.as_uri(), local_dir=checkpoint_dir)


def upload_checkpoint_dir(checkpoint: CheckpointDir, remote_prefix: str) -> CheckpointDir:
    if checkpoint.local_dir is None:
        raise ValueError("upload_checkpoint_dir requires a local checkpoint with a local_dir")

    remote_dir = f"{remote_prefix.rstrip('/')}/{checkpoint.local_dir.name}"
    remote_weights_uri = f"{remote_dir}/{WEIGHTS_FILENAME}"
    write_data(remote_weights_uri, (checkpoint.local_dir / WEIGHTS_FILENAME).read_bytes())

    local_spec = SubmissionPolicySpec.model_validate_json((checkpoint.local_dir / POLICY_SPEC_FILENAME).read_text())
    local_spec.data_path = WEIGHTS_FILENAME
    policy_spec_uri = f"{remote_dir}/{POLICY_SPEC_FILENAME}"
    write_data(policy_spec_uri, local_spec.model_dump_json().encode("utf-8"), content_type="application/json")
    return CheckpointDir(dir_uri=remote_dir)


def resolve_checkpoint_dir(uri: str) -> CheckpointDir:
    parsed = resolve_uri(uri)
    dir_uri = parsed.canonical
    local_dir: Path | None = None

    if parsed.local_path:
        if parsed.local_path.is_file():
            if parsed.local_path.name != POLICY_SPEC_FILENAME:
                raise ValueError("Checkpoint URI must point to a checkpoint directory or policy_spec.json")
            local_dir = parsed.local_path.parent
            dir_uri = local_dir.as_uri()
        else:
            spec_path = parsed.local_path / POLICY_SPEC_FILENAME
            if not spec_path.exists():
                raise ValueError(f"Checkpoint directory missing policy_spec.json: {parsed.local_path}")
            local_dir = parsed.local_path
            dir_uri = parsed.local_path.as_uri()
    else:
        if dir_uri.endswith(POLICY_SPEC_FILENAME):
            dir_uri = dir_uri[: -len(POLICY_SPEC_FILENAME)].rstrip("/")

    if dir_uri.endswith(".zip"):
        raise ValueError("Checkpoint URI must point to a checkpoint directory, not a submission zip")

    return CheckpointDir(dir_uri=dir_uri, local_dir=local_dir)


def load_checkpoint_dir(
    uri: str,
    *,
    cache_dir: Optional[Path] = None,
    remove_downloaded_copy_on_exit: bool = False,
    device: str | None = None,
) -> CheckpointBundle:
    checkpoint = resolve_checkpoint_dir(uri)

    if checkpoint.local_dir is not None:
        local_dir = checkpoint.local_dir
        policy_spec = load_policy_spec_from_local_dir(local_dir, device=device)
        submission_spec = SubmissionPolicySpec.model_validate_json((local_dir / POLICY_SPEC_FILENAME).read_text())
        weights_path = Path(policy_spec.data_path) if policy_spec.data_path else None
        return CheckpointBundle(
            dir_uri=checkpoint.dir_uri,
            local_dir=local_dir,
            policy_spec=policy_spec,
            submission_spec=submission_spec,
            weights_path=weights_path,
        )

    if cache_dir is None:
        cache_dir = DEFAULT_POLICY_CACHE_DIR

    normalized_dir = checkpoint.dir_uri.rstrip("/")
    extraction_root = cache_dir / hashlib.sha256(normalized_dir.encode()).hexdigest()[:16]
    marker_file = extraction_root / ".checkpoint_sync_complete"

    if not marker_file.exists():
        extraction_root.mkdir(parents=True, exist_ok=True)

        spec_uri = f"{normalized_dir}/{POLICY_SPEC_FILENAME}"
        spec_bytes = read(spec_uri)
        submission_spec = SubmissionPolicySpec.model_validate_json(spec_bytes.decode("utf-8"))
        submission_spec, data_path = _normalize_submission_spec_data_path(submission_spec)
        (extraction_root / POLICY_SPEC_FILENAME).write_bytes(submission_spec.model_dump_json().encode("utf-8"))

        if data_path:
            data_uri = f"{normalized_dir}/{data_path.lstrip('/')}"
            local_data_path = extraction_root / data_path
            local_data_path.parent.mkdir(parents=True, exist_ok=True)
            local_data_path.write_bytes(read(data_uri))

        marker_file.touch()

    policy_spec = load_policy_spec_from_local_dir(extraction_root, device=device)
    submission_spec = SubmissionPolicySpec.model_validate_json((extraction_root / POLICY_SPEC_FILENAME).read_text())
    weights_path = Path(policy_spec.data_path) if policy_spec.data_path else None

    if remove_downloaded_copy_on_exit and extraction_root not in _registered_cleanup_dirs:
        _registered_cleanup_dirs.add(extraction_root)
        atexit.register(_cleanup_cache_dir, extraction_root)

    return CheckpointBundle(
        dir_uri=checkpoint.dir_uri,
        local_dir=extraction_root,
        policy_spec=policy_spec,
        submission_spec=submission_spec,
        weights_path=weights_path,
    )


def _cleanup_cache_dir(cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def _normalize_submission_spec_data_path(
    submission_spec: SubmissionPolicySpec,
) -> tuple[SubmissionPolicySpec, Optional[str]]:
    data_path = submission_spec.data_path
    if not data_path:
        return submission_spec, None

    data_path_obj = Path(data_path)
    if data_path_obj.is_absolute():
        normalized = data_path_obj.name
        updated = submission_spec.model_copy()
        updated.data_path = normalized
        return updated, normalized

    return submission_spec, data_path


def write_policy_spec(checkpoint_dir: Path, architecture_spec: str, *, data_path: str = WEIGHTS_FILENAME) -> None:
    submission_spec = SubmissionPolicySpec(
        class_path="mettagrid.policy.checkpoint_policy.CheckpointPolicy",
        data_path=data_path,
        init_kwargs={"architecture_spec": architecture_spec},
    )
    (checkpoint_dir / POLICY_SPEC_FILENAME).write_text(submission_spec.model_dump_json())
