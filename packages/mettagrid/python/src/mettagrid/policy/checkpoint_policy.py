from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.module import load_symbol
from mettagrid.util.uri_resolvers.schemes import checkpoint_filename

WEIGHTS_FILENAME = "weights.safetensors"


def prepare_state_dict_for_save(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    seen_storage: set[int] = set()
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State dict entry '{key}' is not a torch.Tensor")
        value = tensor.detach().cpu()
        data_ptr = value.data_ptr()
        if data_ptr in seen_storage:
            value = value.clone()
        else:
            seen_storage.add(data_ptr)
        result[key] = value
    return result


def _resolve_policy_data_path(path: Path) -> Path:
    if path.is_dir():
        spec_path = path / POLICY_SPEC_FILENAME
        if not spec_path.exists():
            raise FileNotFoundError(f"{POLICY_SPEC_FILENAME} not found in checkpoint directory: {path}")
        submission_spec = SubmissionPolicySpec.model_validate_json(spec_path.read_text())
        if not submission_spec.data_path:
            raise ValueError(f"{POLICY_SPEC_FILENAME} missing data_path in {path}")
        weights_path = path / submission_spec.data_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Policy data path does not exist: {weights_path}")
        return weights_path

    if path.is_file() and path.name != POLICY_SPEC_FILENAME:
        return path

    raise FileNotFoundError(f"Policy data path does not exist: {path}")


def write_policy_spec(checkpoint_dir: Path, architecture_spec: str) -> None:
    spec = SubmissionPolicySpec(
        class_path="mettagrid.policy.checkpoint_policy.CheckpointPolicy",
        data_path=WEIGHTS_FILENAME,
        init_kwargs={"architecture_spec": architecture_spec},
    )
    (checkpoint_dir / POLICY_SPEC_FILENAME).write_text(spec.model_dump_json())


class CheckpointPolicy(MultiAgentPolicy):
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
        class_path = architecture_spec.split("(", 1)[0].strip()
        self._architecture = load_symbol(class_path).from_spec(architecture_spec)
        self._policy = self._architecture.make_policy(policy_env_info).to(self._device)
        self._policy.eval()

    @classmethod
    def from_policy_spec(
        cls,
        policy_env_info: PolicyEnvInterface,
        policy_spec: PolicySpec,
        *,
        device_override: str | None = None,
    ) -> "CheckpointPolicy":
        architecture_spec = policy_spec.init_kwargs.get("architecture_spec")
        if not architecture_spec:
            raise ValueError("policy_spec.json missing init_kwargs.architecture_spec")
        if not policy_spec.data_path:
            raise ValueError("policy_spec.json missing data_path")
        device = device_override or policy_spec.init_kwargs.get("device", "cpu")
        strict = policy_spec.init_kwargs.get("strict", True)
        policy = cls(policy_env_info, architecture_spec=architecture_spec, device=device, strict=strict)
        policy.load_policy_data(policy_spec.data_path)
        return policy

    def load_policy_data(self, policy_data_path: str) -> None:
        weights_blob = _resolve_policy_data_path(Path(policy_data_path).expanduser()).read_bytes()
        state_dict = load_safetensors(weights_blob)
        missing, unexpected = self._policy.load_state_dict(dict(state_dict), strict=False)
        if self._strict and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")
        if hasattr(self._policy, "initialize_to_environment"):
            self._policy.initialize_to_environment(self._policy_env_info, self._device)
        self._policy.eval()

    def save_policy_data(self, policy_data_path: str) -> None:
        target_dir = Path(policy_data_path).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / WEIGHTS_FILENAME).write_bytes(
            save_safetensors(prepare_state_dict_for_save(self._policy.state_dict()))
        )
        write_policy_spec(target_dir, self._architecture_spec)

    @staticmethod
    def write_checkpoint_dir(
        *,
        base_dir: Path,
        run_name: str,
        epoch: int,
        architecture: Any,
        state_dict: Mapping[str, torch.Tensor],
    ) -> Path:
        architecture_spec = architecture if isinstance(architecture, str) else architecture.to_spec()
        checkpoint_dir = (base_dir / checkpoint_filename(run_name, epoch)).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / WEIGHTS_FILENAME).write_bytes(save_safetensors(prepare_state_dict_for_save(state_dict)))
        write_policy_spec(checkpoint_dir, architecture_spec)
        return checkpoint_dir

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "CheckpointPolicy":
        self._policy.eval()
        return self

    @property
    def wrapped_policy(self) -> Any:
        return self._policy
