from __future__ import annotations

import ast
import io
import pickle
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, MutableMapping
from zipfile import BadZipFile

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from hf_metta_policy.modeling_metta_policy import MettaPolicyForRL

from metta.agent.components.component_config import ComponentConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.puffer_policy import _is_puffer_state_dict, load_pufferlib_checkpoint
from metta.rl.training import EnvironmentMetaData
from mettagrid.util.module import load_symbol


def _component_config_to_manifest(component: ComponentConfig) -> dict[str, Any]:
    data = component.model_dump(mode="json")
    data["class_path"] = f"{component.__class__.__module__}.{component.__class__.__qualname__}"
    return data


def _load_component_config(
    data: Any,
    *,
    context: str,
    default_class: type[ComponentConfig] | None = None,
) -> ComponentConfig:
    if isinstance(data, ComponentConfig):
        return data

    if not isinstance(data, Mapping):
        raise TypeError(f"Component config for {context} must be a mapping, got {type(data)!r}")

    class_path = data.get("class_path")
    payload = {key: value for key, value in data.items() if key != "class_path"}

    if not class_path:
        if default_class is None:
            raise ValueError(f"Component config for {context} is missing a class_path attribute")
        return default_class.model_validate(payload)

    component_class = load_symbol(class_path)
    if not isinstance(component_class, type):
        raise TypeError(f"Loaded symbol {class_path} for {context} is not a class")

    # Allow any class that has the required methods, not just ComponentConfig subclasses
    if not (hasattr(component_class, "model_validate") and hasattr(component_class, "model_dump")):
        raise TypeError(f"Loaded symbol {class_path} for {context} does not have required model methods")

    return component_class.model_validate(payload)


def _expr_to_dotted(expr: ast.expr) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return f"{_expr_to_dotted(expr.value)}.{expr.attr}"
    raise ValueError("Expected a dotted name for policy architecture class path")


def _sorted_structure(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sorted_structure(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_structure(item) for item in value]
    return value


def policy_architecture_to_string(architecture: PolicyArchitecture) -> str:
    class_path = f"{architecture.__class__.__module__}.{architecture.__class__.__qualname__}"
    config_data = architecture.model_dump(mode="json")
    config_data.pop("class_path", None)

    if "components" in config_data:
        config_data["components"] = [_component_config_to_manifest(component) for component in architecture.components]

    action_probs_config = getattr(architecture, "action_probs_config", None)
    if action_probs_config is not None:
        config_data["action_probs_config"] = _component_config_to_manifest(action_probs_config)

    if not config_data:
        return class_path

    sorted_config = _sorted_structure(config_data)
    parts = [f"{key}={repr(sorted_config[key])}" for key in sorted(sorted_config)]
    args_repr = ", ".join(parts)
    return f"{class_path}({args_repr})"


def policy_architecture_from_string(spec: str) -> PolicyArchitecture:
    spec = spec.strip()
    if not spec:
        raise ValueError("Policy architecture specification cannot be empty")

    expr = ast.parse(spec, mode="eval").body

    if isinstance(expr, ast.Call):
        class_path = _expr_to_dotted(expr.func)
        kwargs = {}
        for keyword in expr.keywords:
            if keyword.arg is None:
                raise ValueError("Policy architecture arguments must be keyword-based")
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    elif isinstance(expr, (ast.Name, ast.Attribute)):
        class_path = _expr_to_dotted(expr)
        kwargs = {}
    else:
        raise ValueError("Unsupported policy architecture specification format")

    config_class = load_symbol(class_path)
    if not isinstance(config_class, type) or not issubclass(config_class, PolicyArchitecture):
        raise TypeError(f"Loaded symbol {class_path} is not a PolicyArchitecture subclass")

    payload: dict[str, Any] = dict(kwargs)

    default_components: list[ComponentConfig] = []
    default_action_probs: ComponentConfig | None = None
    try:
        default_instance = config_class()
        default_components = list(getattr(default_instance, "components", []) or [])
        default_action_probs = getattr(default_instance, "action_probs_config", None)
    except Exception:
        pass

    components_data = payload.get("components")
    if components_data is not None:
        if not isinstance(components_data, list):
            raise TypeError("Policy architecture components must be provided as a list")
        payload["components"] = [
            _load_component_config(
                component,
                context=f"component[{index}]",
                default_class=(default_components[index].__class__ if index < len(default_components) else None),
            )
            for index, component in enumerate(components_data)
        ]

    action_probs_data = payload.get("action_probs_config")
    if action_probs_data is not None:
        default_class = default_action_probs.__class__ if default_action_probs is not None else None
        payload["action_probs_config"] = _load_component_config(
            action_probs_data,
            context="action_probs_config",
            default_class=default_class,
        )

    architecture = config_class.model_validate(payload)
    if not isinstance(architecture, PolicyArchitecture):
        raise TypeError("Deserialized object is not a PolicyArchitecture")
    return architecture


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

    HF_ARTIFACT_PATH: ClassVar[Path] = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "hf"
        / "c65ada04bb1f477d944b84494ffa64ee86045025"
    )

    @classmethod
    def from_hardcoded_hf(cls) -> "PolicyArtifact":
        policy = load_policy_from_hf_artifact(cls.HF_ARTIFACT_PATH)
        return cls(policy=policy)

    def __post_init__(self) -> None:
        has_arch = self.policy_architecture is not None
        has_state = self.state_dict is not None
        has_policy = self.policy is not None

        valid_combo = (has_state and has_arch and not has_policy) or (has_policy and not has_state and not has_arch)

        if not valid_combo:
            msg = "PolicyArtifact must contain either (policy) or (state_dict + policy_architecture)."
            raise ValueError(msg)

        if has_state and not isinstance(self.state_dict, MutableMapping):
            msg = "state_dict must be a mutable mapping of parameter tensors"
            raise TypeError(msg)

    def instantiate(
        self,
        env_metadata: EnvironmentMetaData,
        device: torch.device,
        *,
        strict: bool = True,
    ) -> Policy:
        if self.state_dict is not None and self.policy_architecture is not None:
            policy = self.policy_architecture.make_policy(env_metadata)
            policy = policy.to(device)

            if hasattr(policy, "initialize_to_environment"):
                policy.initialize_to_environment(env_metadata, device)

            ordered_state = OrderedDict(self.state_dict.items())
            missing, unexpected = policy.load_state_dict(ordered_state, strict=strict)
            if strict and (missing or unexpected):
                msg = f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}"
                raise RuntimeError(msg)
            self.policy = policy
            self.state_dict = None
            return policy

        if self.policy is not None:
            self.policy = self.policy.to(device)
            return self.policy

        msg = "Cannot instantiate artifact without weights/architecture or policy"
        raise ValueError(msg)


def save_policy_artifact_safetensors(
    path: str | Path,
    *,
    policy_architecture: PolicyArchitecture,
    state_dict: Mapping[str, torch.Tensor],
    detach_buffers: bool = True,
) -> PolicyArtifact:
    """Persist weights + architecture using the safetensors format."""
    return _save_policy_artifact(
        path,
        policy_architecture=policy_architecture,
        state_dict=state_dict,
        detach_buffers=detach_buffers,
    )


def save_policy_artifact_pt(
    path: str | Path,
    *,
    policy: Policy,
) -> PolicyArtifact:
    """Persist a policy object with torch.save (.pt)."""
    return _save_policy_artifact(path, policy=policy, include_policy=True)


def _save_policy_artifact(
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
        if has_state_input:
            msg = "include_policy=True cannot be combined with weights/state_dict"
            raise ValueError(msg)
        buffer = io.BytesIO()
        torch.save(policy, buffer)
        policy_payload = buffer.getvalue()

    # Atomic save: write to temporary file first, then move to final destination
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

        try:
            with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                if artifact_state is not None and policy_architecture is not None:
                    weights_blob = save_safetensors(artifact_state)
                    archive.writestr("weights.safetensors", weights_blob)
                    archive.writestr(
                        "modelarchitecture.txt",
                        policy_architecture_to_string(policy_architecture),
                    )

                if policy_payload is not None:
                    archive.writestr("policy.pt", policy_payload)

            # Atomic move: this operation is atomic on most filesystems
            temp_path.replace(output_path)

        except Exception:
            # Clean up temporary file on error
            temp_path.unlink(missing_ok=True)
            raise

    return PolicyArtifact(
        policy_architecture=policy_architecture if artifact_state is not None else None,
        state_dict=artifact_state,
        policy=policy if include_policy else None,
    )


def load_policy_artifact(path: str | Path, is_pt_file: bool = False) -> PolicyArtifact:
    input_path = Path(path)
    if not input_path.exists():
        msg = f"Policy artifact not found: {input_path}"
        raise FileNotFoundError(msg)

    if is_pt_file or input_path.suffix == ".pt":
        try:
            legacy_payload = torch.load(input_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            raise
        except (pickle.UnpicklingError, RuntimeError, OSError, TypeError, BadZipFile) as err:
            raise FileNotFoundError(f"Invalid or corrupted checkpoint file: {input_path}") from err

        if _is_puffer_state_dict(legacy_payload):
            policy = load_pufferlib_checkpoint(legacy_payload, device="cpu")
            return PolicyArtifact(policy=policy)

        return PolicyArtifact(policy=legacy_payload)

    architecture: PolicyArchitecture | None = None
    state_dict: MutableMapping[str, torch.Tensor] | None = None
    policy: Policy | None = None

    with zipfile.ZipFile(input_path, mode="r") as archive:
        names = set(archive.namelist())

        if "modelarchitecture.txt" in names and "weights.safetensors" in names:
            architecture_blob = archive.read("modelarchitecture.txt").decode("utf-8")
            architecture = policy_architecture_from_string(architecture_blob)

            weights_blob = archive.read("weights.safetensors")
            loaded_state = load_safetensors(weights_blob)
            if not isinstance(loaded_state, MutableMapping):
                msg = "Loaded safetensors state_dict is not a mutable mapping"
                raise TypeError(msg)
            state_dict = loaded_state

        elif "policy.pt" in names:
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


def load_policy_from_hf_artifact(path: str | Path) -> Policy:
    """Load a policy that was exported via the HF interoperability helper."""

    model = MettaPolicyForRL.from_pretrained(Path(path))
    policy = getattr(model, "policy", None)
    if policy is None:
        raise ValueError(f"Hugging Face artifact at {path} did not expose a policy module")
    if not isinstance(policy, Policy):
        raise TypeError(f"Loaded module from {path} is not a Policy instance: {type(policy)!r}")
    return policy
