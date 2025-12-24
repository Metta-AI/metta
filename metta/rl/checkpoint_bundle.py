from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping, Tuple

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.uri_resolvers.schemes import checkpoint_filename

WEIGHTS_FILENAME = "weights.safetensors"


def prepare_state_dict_for_save(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    seen_storage: set[int] = set()
    for key, tensor in state_dict.items():
        value = tensor.detach().cpu()
        data_ptr = value.data_ptr()
        if data_ptr in seen_storage:
            value = value.clone()
        else:
            seen_storage.add(data_ptr)
        result[key] = value
    return result


def write_checkpoint_dir(
    *,
    base_dir: Path,
    run_name: str,
    epoch: int,
    policy_class_path: str,
    architecture_spec: str,
    state_dict: Mapping[str, torch.Tensor],
) -> Path:
    checkpoint_dir = (base_dir / checkpoint_filename(run_name, epoch)).expanduser().resolve()
    write_checkpoint_bundle(
        checkpoint_dir,
        policy_class_path=policy_class_path,
        architecture_spec=architecture_spec,
        state_dict=state_dict,
    )
    return checkpoint_dir


def write_checkpoint_bundle(
    checkpoint_dir: Path,
    *,
    policy_class_path: str,
    architecture_spec: str,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_blob = save_safetensors(prepare_state_dict_for_save(state_dict))
    _write_file_atomic(checkpoint_dir / WEIGHTS_FILENAME, weights_blob)
    spec = SubmissionPolicySpec(
        class_path=policy_class_path,
        data_path=WEIGHTS_FILENAME,
        init_kwargs={"architecture_spec": architecture_spec},
    )
    _write_file_atomic(checkpoint_dir / POLICY_SPEC_FILENAME, spec.model_dump_json().encode("utf-8"))


def load_checkpoint_state(policy_spec: PolicySpec) -> Tuple[str, dict[str, torch.Tensor]]:
    architecture_spec = policy_spec.init_kwargs["architecture_spec"]
    path = Path(policy_spec.data_path).expanduser()
    state_dict = dict(load_safetensors(path.read_bytes()))
    return architecture_spec, state_dict


def _write_file_atomic(path: Path, data: bytes) -> None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(data)
        tmp_path.replace(path)
        tmp_path = None
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
