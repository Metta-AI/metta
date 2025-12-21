from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.architecture_spec import architecture_from_spec, architecture_spec_from_value
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.uri_resolvers.schemes import checkpoint_filename

WEIGHTS_FILENAME = "weights.safetensors"


def _join_uri(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child}"


def prepare_state_dict_for_save(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def _resolve_policy_data_path(policy_data_path: Path) -> Path:
    if policy_data_path.is_dir():
        spec_path = policy_data_path / POLICY_SPEC_FILENAME
        if not spec_path.exists():
            raise FileNotFoundError(f"{POLICY_SPEC_FILENAME} not found in checkpoint directory: {policy_data_path}")
        submission_spec = SubmissionPolicySpec.model_validate_json(spec_path.read_text())
        if not submission_spec.data_path:
            raise ValueError(f"{POLICY_SPEC_FILENAME} missing data_path in {policy_data_path}")
        weights_path = policy_data_path / submission_spec.data_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Policy data path does not exist: {weights_path}")
        return weights_path

    if policy_data_path.is_file() and policy_data_path.name == POLICY_SPEC_FILENAME:
        submission_spec = SubmissionPolicySpec.model_validate_json(policy_data_path.read_text())
        if not submission_spec.data_path:
            raise ValueError(f"{POLICY_SPEC_FILENAME} missing data_path in {policy_data_path}")
        weights_path = policy_data_path.parent / submission_spec.data_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Policy data path does not exist: {weights_path}")
        return weights_path

    return policy_data_path


def _write_policy_spec(checkpoint_dir: Path, architecture_spec: str, *, data_path: str = WEIGHTS_FILENAME) -> None:
    submission_spec = SubmissionPolicySpec(
        class_path="mettagrid.policy.checkpoint_policy.CheckpointPolicy",
        data_path=data_path,
        init_kwargs={"architecture_spec": architecture_spec},
    )
    (checkpoint_dir / POLICY_SPEC_FILENAME).write_text(submission_spec.model_dump_json())


@dataclass(frozen=True)
class CheckpointDir:
    """Checkpoint directory containing policy_spec.json and weights.safetensors."""

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


class CheckpointPolicy(MultiAgentPolicy):
    """Policy implementation that loads weights from a policy_spec directory.

    Expected PolicySpec fields:
      - init_kwargs.architecture_spec: serialized architecture (via to_spec())
      - data_path: safetensors weights file (resolved to local path by PolicySpec loader)
    """

    short_names = ["checkpoint"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        architecture_spec: str,
        device: str = "cpu",
        strict: bool = True,
    ):
        super().__init__(policy_env_info, device=device)
        self._strict = strict
        self._device = torch.device(device)
        self._policy_env_info = policy_env_info
        self._architecture_spec = architecture_spec
        self._architecture = architecture_from_spec(architecture_spec)
        self._policy = self._architecture.make_policy(policy_env_info).to(self._device)
        self._policy.eval()

    def load_policy_data(self, policy_data_path: str) -> None:
        weights_path = _resolve_policy_data_path(Path(policy_data_path).expanduser())
        weights_blob = weights_path.read_bytes()
        state_dict = load_safetensors(weights_blob)
        missing, unexpected = self._policy.load_state_dict(dict(state_dict), strict=self._strict)
        if self._strict and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

        if hasattr(self._policy, "initialize_to_environment"):
            self._policy.initialize_to_environment(self._policy_env_info, self._device)
        self._policy.eval()

    def save_policy_data(self, policy_data_path: str) -> None:
        target_path = Path(policy_data_path).expanduser()
        if target_path.is_dir() or target_path.suffix == "":
            target_path.mkdir(parents=True, exist_ok=True)
            weights_path = target_path / WEIGHTS_FILENAME
            prepared_state = prepare_state_dict_for_save(self._policy.state_dict())
            weights_path.write_bytes(save_safetensors(dict(prepared_state)))
            _write_policy_spec(target_path, self._architecture_spec)
            return

        prepared_state = prepare_state_dict_for_save(self._policy.state_dict())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(save_safetensors(dict(prepared_state)))
        _write_policy_spec(target_path.parent, self._architecture_spec, data_path=target_path.name)

    @staticmethod
    def write_checkpoint_dir(
        *,
        base_dir: Path,
        run_name: str,
        epoch: int,
        architecture: Any,
        state_dict: Mapping[str, torch.Tensor],
    ) -> CheckpointDir:
        """Write a checkpoint directory to disk and return its metadata."""
        architecture_spec = architecture_spec_from_value(architecture)
        checkpoint_dir = (base_dir / checkpoint_filename(run_name, epoch)).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        weights_path = checkpoint_dir / WEIGHTS_FILENAME
        prepared_state = prepare_state_dict_for_save(state_dict)
        weights_path.write_bytes(save_safetensors(dict(prepared_state)))
        _write_policy_spec(checkpoint_dir, architecture_spec)

        return CheckpointDir(dir_uri=checkpoint_dir.as_uri(), local_dir=checkpoint_dir)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "CheckpointPolicy":
        self._policy.eval()
        return self

    @property
    def wrapped_policy(self) -> Any:
        return self._policy
