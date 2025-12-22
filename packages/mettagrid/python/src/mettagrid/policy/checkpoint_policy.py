from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mettagrid.policy.architecture_spec import architecture_from_spec
from mettagrid.policy.checkpoint_io import (
    WEIGHTS_FILENAME,
    CheckpointDir,
    load_checkpoint_dir,
    prepare_state_dict_for_save,
    write_checkpoint_dir as write_checkpoint_dir_io,
    write_policy_spec,
)
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


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
        if "://" in policy_data_path:
            bundle = load_checkpoint_dir(policy_data_path)
            if bundle.weights_path is None:
                raise ValueError("Checkpoint policy_spec.json missing data_path")
            weights_blob = bundle.weights_path.read_bytes()
        else:
            local_path = Path(policy_data_path).expanduser()
            if local_path.is_file() and local_path.name != "policy_spec.json":
                weights_blob = local_path.read_bytes()
            else:
                bundle = load_checkpoint_dir(str(local_path))
                if bundle.weights_path is None:
                    raise ValueError("Checkpoint policy_spec.json missing data_path")
                weights_blob = bundle.weights_path.read_bytes()

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
            write_policy_spec(target_path, self._architecture_spec)
            return

        prepared_state = prepare_state_dict_for_save(self._policy.state_dict())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(save_safetensors(dict(prepared_state)))
        write_policy_spec(target_path.parent, self._architecture_spec, data_path=target_path.name)

    @staticmethod
    def write_checkpoint_dir(
        *,
        base_dir: Path,
        run_name: str,
        epoch: int,
        architecture: Any,
        state_dict: Mapping[str, torch.Tensor],
    ) -> CheckpointDir:
        return write_checkpoint_dir_io(
            base_dir=base_dir,
            run_name=run_name,
            epoch=epoch,
            architecture=architecture,
            state_dict=state_dict,
        )

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return self._policy.agent_policy(agent_id)

    def eval(self) -> "CheckpointPolicy":
        self._policy.eval()
        return self

    @property
    def wrapped_policy(self) -> Any:
        return self._policy
