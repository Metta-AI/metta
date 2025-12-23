from __future__ import annotations

import tempfile
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


class CheckpointPolicy(MultiAgentPolicy):
    CLASS_PATH = "mettagrid.policy.checkpoint_policy.CheckpointPolicy"
    short_names = ["checkpoint"]
    WEIGHTS_FILENAME = "weights.safetensors"

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        architecture_spec: str,
        device: str = "cpu",
    ):
        super().__init__(policy_env_info, device=device)
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
    ) -> "CheckpointPolicy":
        architecture_spec = policy_spec.init_kwargs["architecture_spec"]
        device = policy_spec.init_kwargs.get("device", "cpu")
        policy = cls(policy_env_info, architecture_spec=architecture_spec, device=device)
        policy.load_policy_data(policy_spec.data_path)
        return policy

    def load_policy_data(self, policy_data_path: str) -> None:
        path = Path(policy_data_path).expanduser()
        state_dict = load_safetensors(path.read_bytes())
        self._policy.load_state_dict(dict(state_dict))
        if hasattr(self._policy, "initialize_to_environment"):
            self._policy.initialize_to_environment(self._policy_env_info, self._device)
        self._policy.eval()

    def save_policy_data(self, policy_data_path: str) -> None:
        _write_checkpoint_bundle(
            Path(policy_data_path).expanduser(),
            architecture_spec=self._architecture_spec,
            state_dict=self._policy.state_dict(),
        )

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
        _write_checkpoint_bundle(
            checkpoint_dir,
            architecture_spec=architecture_spec,
            state_dict=state_dict,
        )
        return checkpoint_dir

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "CheckpointPolicy":
        self._policy.eval()
        return self

    @property
    def wrapped_policy(self) -> Any:
        return self._policy

    @property
    def architecture_spec(self) -> str:
        return self._architecture_spec


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
        tmp_path = None  # Mark as successfully moved
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def _write_checkpoint_bundle(
    checkpoint_dir: Path,
    *,
    architecture_spec: str,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_blob = save_safetensors(prepare_state_dict_for_save(state_dict))
    _write_file_atomic(checkpoint_dir / CheckpointPolicy.WEIGHTS_FILENAME, weights_blob)
    spec = SubmissionPolicySpec(
        class_path=CheckpointPolicy.CLASS_PATH,
        data_path=CheckpointPolicy.WEIGHTS_FILENAME,
        init_kwargs={"architecture_spec": architecture_spec},
    )
    _write_file_atomic(checkpoint_dir / POLICY_SPEC_FILENAME, spec.model_dump_json().encode("utf-8"))
