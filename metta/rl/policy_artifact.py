from __future__ import annotations

import io
import json
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.puffer_policy import _is_puffer_state_dict, load_pufferlib_checkpoint
from metta.rl.training import EnvironmentMetaData
from mettagrid.util.module import load_symbol


def _architecture_class_path(policy_architecture: PolicyArchitecture) -> str:
    architecture_class = policy_architecture.__class__
    return f"{architecture_class.__module__}.{architecture_class.__qualname__}"


def _serialize_policy_architecture(policy_architecture: PolicyArchitecture) -> bytes:
    payload = {
        "config_class": _architecture_class_path(policy_architecture),
        "config": policy_architecture.model_dump(mode="json"),
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def _deserialize_policy_architecture(blob: bytes) -> PolicyArchitecture:
    data = json.loads(blob.decode("utf-8"))

    config_class_path = data.get("config_class")
    if not config_class_path:
        msg = "Missing 'config_class' in policy architecture payload"
        raise ValueError(msg)

    config_data = data.get("config")
    if config_data is None:
        msg = "Missing 'config' payload in policy architecture"
        raise ValueError(msg)

    config_class = load_symbol(config_class_path)
    if not isinstance(config_class, type) or not issubclass(config_class, PolicyArchitecture):
        msg = f"Config class {config_class_path} is not a PolicyArchitecture"
        raise TypeError(msg)

    return config_class.model_validate(config_data)


def _to_safetensors_state_dict(
    state_dict: Mapping[str, torch.Tensor], detach_buffers: bool
) -> MutableMapping[str, torch.Tensor]:
    """
    according to codex:
    state_dict() technically gives you an OrderedDict[str, Tensor], but it often contains parameter Tensors that may
    live on GPU and can still be tied to the computation graph (grad history or shared buffers). safetensors.save_file
    insists on plain CPU tensors with no live autograd ties.

    _to_safetensors_state_dict strips any gradients (detach()) and moves
    everything to CPU before handing that mapping to safetensors, keeping the order stable for deterministic saves.
    Without that preprocessing a GPU-only or gradient‑tracking tensor would make save_file barf.
    """

    ordered: MutableMapping[str, torch.Tensor] = OrderedDict()
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            msg = f"State dict entry '{key}' is not a torch.Tensor"
            raise TypeError(msg)
        value = tensor.detach() if detach_buffers else tensor
        ordered[key] = value.cpu()
    return ordered


@dataclass
class PolicyArtifact:
    policy_architecture: PolicyArchitecture | None = None
    state_dict: MutableMapping[str, torch.Tensor] | None = None
    policy: Policy | None = None

    def __post_init__(self) -> None:
        has_arch = self.policy_architecture is not None
        has_state = self.state_dict is not None
        has_policy = self.policy is not None

        valid_combo = (
            (has_state and has_arch and not has_policy)
            or (has_policy and not has_state and not has_arch)
            or (has_state and has_arch and has_policy)
        )

        if not valid_combo:
            msg = (
                "PolicyArtifact must contain either (policy), "
                "(state_dict + policy_architecture), or (state_dict + policy_architecture + policy)"
            )
            raise ValueError(msg)

        if has_state and not isinstance(self.state_dict, MutableMapping):
            msg = "state_dict must be a mutable mapping of parameter tensors"
            raise TypeError(msg)

    def instantiate(self, env_metadata: EnvironmentMetaData, strict: bool = True) -> Policy:
        if self.state_dict is not None and self.policy_architecture is not None:
            policy = self.policy_architecture.make_policy(env_metadata)
            ordered_state = OrderedDict(self.state_dict.items())
            missing, unexpected = policy.load_state_dict(ordered_state, strict=strict)
            if strict and (missing or unexpected):
                msg = f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}"
                raise RuntimeError(msg)
            self.policy = policy
            self.state_dict = None
            return policy

        if self.policy is not None:
            return self.policy

        msg = "Cannot instantiate artifact without weights/architecture or policy"
        raise ValueError(msg)


def save_policy_artifact(
    path: str | Path,
    *,
    policy: Policy | None = None,
    policy_architecture: PolicyArchitecture | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    include_policy: bool = False,
    detach_buffers: bool = True,
) -> PolicyArtifact:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_state_input = state_dict is not None
    if not has_state_input and policy is not None and policy_architecture is not None:
        state_dict = policy.state_dict()

    has_state_input = state_dict is not None

    if has_state_input and policy_architecture is None:
        msg = "policy_architecture is required when saving weights"
        raise ValueError(msg)

    if not has_state_input and not include_policy:
        msg = "Saving requires weights/architecture or include_policy=True with a policy"
        raise ValueError(msg)

    artifact_state: MutableMapping[str, torch.Tensor] | None = None
    if has_state_input:
        artifact_state = _to_safetensors_state_dict(state_dict or {}, detach_buffers)

    policy_payload: bytes | None = None
    if include_policy:
        if policy is None:
            msg = "include_policy=True requires a policy instance"
            raise ValueError(msg)
        buffer = io.BytesIO()
        torch.save(policy, buffer)
        policy_payload = buffer.getvalue()

    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if artifact_state is not None and policy_architecture is not None:
            weights_blob = save_safetensors(artifact_state)
            archive.writestr("weights.safetensors", weights_blob)
            archive.writestr("modelarchitecture.json", _serialize_policy_architecture(policy_architecture))

        if policy_payload is not None:
            archive.writestr("policy.pt", policy_payload)

    return PolicyArtifact(
        policy_architecture=policy_architecture if artifact_state is not None else None,
        state_dict=artifact_state,
        policy=policy if include_policy else None,
    )


def load_policy_artifact(path: str | Path) -> PolicyArtifact:
    input_path = Path(path)
    if not input_path.exists():
        msg = f"Policy artifact not found: {input_path}"
        raise FileNotFoundError(msg)

    architecture: PolicyArchitecture | None = None
    state_dict: MutableMapping[str, torch.Tensor] | None = None
    policy: Policy | None = None

    with zipfile.ZipFile(input_path, mode="r") as archive:
        names = set(archive.namelist())

        if "modelarchitecture.json" in names and "weights.safetensors" in names:
            architecture_blob = archive.read("modelarchitecture.json")
            architecture = _deserialize_policy_architecture(architecture_blob)

            weights_blob = archive.read("weights.safetensors")
            loaded_state = load_safetensors(weights_blob)
            if not isinstance(loaded_state, MutableMapping):
                msg = "Loaded safetensors state_dict is not a mutable mapping"
                raise TypeError(msg)
            state_dict = loaded_state

        if "policy.pt" in names:
            buffer = io.BytesIO(archive.read("policy.pt"))
            loaded_policy = torch.load(buffer, map_location="cpu", weights_only=False)

            if _is_puffer_state_dict(loaded_policy):
                policy = load_pufferlib_checkpoint(loaded_policy, device="cpu")
            else:
                if not isinstance(loaded_policy, Policy):
                    msg = "Loaded policy payload is not a Policy instance"
                    raise TypeError(msg)
                policy = loaded_policy

    if architecture is None and state_dict is None and policy is None:
        msg = f"Policy artifact {input_path} contained no usable payload"
        raise ValueError(msg)

    return PolicyArtifact(policy_architecture=architecture, state_dict=state_dict, policy=policy)
