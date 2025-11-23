from __future__ import annotations

import ast
import io
import logging
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional
from zipfile import BadZipFile

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from metta.agent.components.component_config import ComponentConfig
from metta.agent.mocks import MockAgent
from metta.agent.policy import Policy, PolicyArchitecture
from metta.common.util.file import local_copy
from metta.common.util.uri import ParsedURI
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)


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
    seen_storage: dict[int, str] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            msg = f"State dict entry '{key}' is not a torch.Tensor"
            raise TypeError(msg)
        value = tensor.detach() if detach_buffers else tensor
        value_cpu = value.cpu()
        data_ptr = value_cpu.data_ptr()
        if data_ptr in seen_storage:
            # safetensors forbids tensors that alias the same storage. Clone to materialize
            # an independent copy while preserving dtype/shape metadata.
            value_cpu = value_cpu.clone()
        else:
            seen_storage[data_ptr] = key

        ordered[key] = value_cpu
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

        valid_combo = (has_state and not has_policy) or (has_policy and not has_state and not has_arch)

        if not valid_combo:
            msg = (
                "PolicyArtifact must contain either (policy) or (state_dict). "
                "When using state_dict, provide policy_architecture before calling instantiate()."
            )
            raise ValueError(msg)

        if has_state and not isinstance(self.state_dict, MutableMapping):
            msg = "state_dict must be a mutable mapping of parameter tensors"
            raise TypeError(msg)

    def instantiate(
        self,
        policy_env_info: PolicyEnvInterface,
        device: torch.device,
        *,
        strict: bool = True,
    ) -> Policy:
        if self.state_dict is not None:
            if self.policy_architecture is None:
                msg = "policy_architecture is required to instantiate weights-only artifacts"
                raise ValueError(msg)
            policy = self.policy_architecture.make_policy(policy_env_info)
            policy = policy.to(device)

            if hasattr(policy, "initialize_to_environment"):
                policy.initialize_to_environment(policy_env_info, device)

            ordered_state = OrderedDict(self.state_dict.items())

            # If saved without the policy prefix, add it for AutoBuilder-style policies
            if ordered_state and not any(k.startswith("_sequential_network") for k in ordered_state):
                model_keys = policy.state_dict().keys()
                if any(k.startswith("_sequential_network.module.") for k in model_keys):
                    ordered_state = OrderedDict(
                        (f"_sequential_network.module.{k}", v) for k, v in ordered_state.items()
                    )

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


def save_policy_artifact(
    path: str | Path,
    *,
    policy_architecture: PolicyArchitecture,
    state_dict: Mapping[str, torch.Tensor],
    detach_buffers: bool = True,
) -> PolicyArtifact:
    """Persist weights + architecture using the safetensors format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_state = _to_safetensors_state_dict(state_dict, detach_buffers)

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

        try:
            with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                weights_blob = save_safetensors(artifact_state)
                archive.writestr("weights.safetensors", weights_blob)
                archive.writestr(
                    "modelarchitecture.txt",
                    policy_architecture_to_string(policy_architecture),
                )

            temp_path.replace(output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    return PolicyArtifact(policy_architecture=policy_architecture, state_dict=artifact_state)


def _normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
    """Strip common DDP prefixes and collapse duplicate '.module.' segments.

    Handles checkpoints saved from distributed wrappers where keys are prefixed with
    'module.' and cases where nested modules introduce '.module.module.' sequences.
    """

    normalized: MutableMapping[str, torch.Tensor] = OrderedDict()
    has_global_ddp_prefix = state_dict and all(key.startswith("module.") for key in state_dict)
    for key, value in state_dict.items():
        new_key = key
        if has_global_ddp_prefix and new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        while ".module.module." in new_key:
            new_key = new_key.replace(".module.module.", ".module.")
        normalized[new_key] = value
    return normalized


def _try_load_pufferlib_checkpoint(payload: object) -> Policy | None:
    """Best-effort pufferlib compatibility loader."""
    try:
        from metta.rl.puffer_policy import _is_puffer_state_dict, load_pufferlib_checkpoint
    except Exception:
        return None

    if _is_puffer_state_dict(payload):
        return load_pufferlib_checkpoint(payload)
    return None


def _artifact_from_payload(payload: object, *, source: str) -> PolicyArtifact:
    if isinstance(payload, Policy):
        return PolicyArtifact(policy=payload)

    if not isinstance(payload, Mapping):
        raise TypeError(f"{source} must contain a state_dict mapping, got {type(payload)}")

    puffer_policy = _try_load_pufferlib_checkpoint(payload)
    if puffer_policy is not None:
        return PolicyArtifact(policy=puffer_policy)

    # Backwards compat: handle wrapped dicts (old: {"state_dict": {...}}, new: {...})
    state_dict = payload.get("state_dict") or payload.get("weights") or payload
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Wrapped state_dict must be a mapping, got {type(state_dict)}")

    # Extract architecture if present (legacy keys)
    policy_architecture = None
    if "state_dict" in payload or "weights" in payload:
        arch_value = (
            payload.get("policy_architecture")
            or payload.get("policy_architecture_spec")
            or payload.get("policy_architecture_str")
        )
        if isinstance(arch_value, str):
            policy_architecture = policy_architecture_from_string(arch_value)
        elif hasattr(arch_value, "__class__") and arch_value.__class__.__name__ == "PolicyArchitecture":
            policy_architecture = arch_value

    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state_dict.items()):
        raise TypeError(f"{source} must contain string→Tensor mapping")

    return PolicyArtifact(
        policy_architecture=policy_architecture,
        state_dict=_normalize_state_dict_keys(state_dict),
    )


def _load_mpt_artifact(path: Path) -> PolicyArtifact:
    """Load modern .mpt format: safetensors + architecture."""
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            names = set(archive.namelist())

            # Backwards compat: old .mpt files are PyTorch ZIP format (data.pkl)
            if "weights.safetensors" not in names:
                if any("data.pkl" in name for name in names):
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    # Route legacy PyTorch ZIP payloads through the generic loader so we can
                    # recover embedded metadata (e.g., policy_architecture_{spec,str}) when
                    # present instead of treating them as raw weights only.
                    return _artifact_from_payload(payload, source=".mpt data.pkl")
                if "policy.pt" in names:
                    try:
                        with archive.open("policy.pt") as payload_file:
                            payload = torch.load(
                                io.BytesIO(payload_file.read()),
                                map_location="cpu",
                                weights_only=False,
                            )
                    except Exception as exc:
                        raise ValueError(f"Failed to load policy.pt from .mpt file: {path}") from exc
                    return _artifact_from_payload(payload, source=".mpt policy.pt")
                raise ValueError(f".mpt file missing weights.safetensors: {path}")

            if "modelarchitecture.txt" not in names:
                raise ValueError(f".mpt file missing modelarchitecture.txt: {path}")

            state_dict = load_safetensors(archive.read("weights.safetensors"))
            architecture = policy_architecture_from_string(archive.read("modelarchitecture.txt").decode("utf-8"))

            return PolicyArtifact(
                policy_architecture=architecture,
                state_dict=_normalize_state_dict_keys(state_dict),
            )
    except BadZipFile as e:
        raise ValueError(f"Invalid .mpt file (not a valid ZIP archive): {path}") from e


def _load_pt_artifact(path: Path) -> PolicyArtifact:
    """Load simple .pt format: raw state_dict pickle.

    Used for cogames-trained policies and backward compatibility.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load .pt file: {path}") from e

    return _artifact_from_payload(payload, source=".pt file")


def _extract_run_and_epoch(path: Path) -> tuple[str, int] | None:
    """Infer run name and epoch from a checkpoint path."""

    stem = path.stem

    if ":v" in stem:
        run_name, suffix = stem.rsplit(":v", 1)
        if run_name and suffix.isdigit():
            return (run_name, int(suffix))
    return None


def _boto3_client():
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "boto3 is required for S3 checkpoint handling; install the RL extras or add boto3"
        ) from exc
    return boto3.client("s3")


def _get_all_checkpoints(uri: str) -> list[dict[str, Any]]:
    """List checkpoint metadata for file:// or s3:// URIs."""

    parsed = ParsedURI.parse(uri)
    if parsed.scheme == "file" and parsed.local_path:
        checkpoint_files = [ckpt for ckpt in parsed.local_path.glob("*.mpt") if ckpt.stem]
    elif parsed.scheme == "s3" and parsed.bucket:
        s3_client = _boto3_client()
        prefix = parsed.key or ""
        response = s3_client.list_objects_v2(Bucket=parsed.bucket, Prefix=prefix)

        if response["KeyCount"] == 0:
            return []

        checkpoint_files: list[Path] = [Path(obj["Key"]) for obj in response["Contents"] if obj["Key"].endswith(".mpt")]
    else:
        raise ValueError(f"Cannot get checkpoints from uri: {uri}")

    checkpoint_metadata: list[dict[str, Any]] = []
    for path in checkpoint_files:
        run_and_epoch = _extract_run_and_epoch(path)
        if run_and_epoch:
            path_uri = uri.rstrip("/") + "/" + path.name
            checkpoint_metadata.append(
                {
                    "run_name": run_and_epoch[0],
                    "epoch": run_and_epoch[1],
                    "uri": path_uri,
                }
            )

    return checkpoint_metadata


def _latest_checkpoint_uri(uri: str) -> str:
    checkpoints = _get_all_checkpoints(uri)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files in {uri}")
    latest = max(checkpoints, key=lambda p: p["epoch"])
    return latest["uri"]


def _normalize_policy_uri(uri: str) -> str:
    """Convert paths to file:// URIs and resolve :latest selectors."""

    parsed = ParsedURI.parse(uri)
    if uri.endswith(":latest"):
        base_uri = uri[:-7]
        return _latest_checkpoint_uri(base_uri)
    return parsed.canonical


def load_policy_artifact(path_or_uri: str | Path) -> PolicyArtifact:
    """Load a policy artifact from .mpt/.pt or URI (file://, s3://, :latest, mock://)."""
    if isinstance(path_or_uri, Path):
        input_path = path_or_uri
    else:
        if "://" not in path_or_uri and not str(path_or_uri).endswith(":latest"):
            input_path = Path(path_or_uri)
        else:
            input_path = None

    if input_path is not None:
        if input_path.exists():
            if input_path.suffix == ".mpt":
                return _load_mpt_artifact(input_path)
            if input_path.suffix == ".pt":
                return _load_pt_artifact(input_path)
            raise ValueError(f"Unsupported checkpoint extension: {input_path.suffix}. Expected .mpt or .pt")
        else:
            raise FileNotFoundError(f"Policy artifact not found: {input_path}")

    uri = str(path_or_uri)
    if uri.startswith(("http://", "https://", "ftp://", "gs://")):
        raise ValueError(f"Invalid URI: {uri}")

    normalized_uri = _normalize_policy_uri(uri)
    parsed = ParsedURI.parse(normalized_uri)

    if parsed.scheme == "file" and parsed.local_path is not None:
        path = parsed.local_path
        if path.is_dir():
            latest_uri = _latest_checkpoint_uri(normalized_uri)
            parsed_latest = ParsedURI.parse(latest_uri)
            local_path = parsed_latest.local_path
            if local_path is None:
                raise FileNotFoundError(f"No checkpoint files in {normalized_uri}")
            return load_policy_artifact(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return load_policy_artifact(path)

    if parsed.scheme == "s3":
        with local_copy(parsed.canonical) as local_path:
            return load_policy_artifact(local_path)

    if parsed.scheme == "mock":
        return PolicyArtifact(policy=MockAgent())

    raise ValueError(f"Invalid URI: {uri}")


def policy_spec_from_uri(
    uri: str,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
    display_name: str | None = None,
    policy_architecture: PolicyArchitecture | None = None,
    class_path: str | None = None,
) -> PolicySpec:
    """Construct a PolicySpec for a checkpoint URI, with legacy metadata support."""
    normalized_uri = _normalize_policy_uri(uri)
    parsed_uri = ParsedURI.parse(normalized_uri)

    artifact: Optional[PolicyArtifact] = None
    embedded_policy_class_path: Optional[str] = None
    if policy_architecture is None:
        artifact = load_policy_artifact(normalized_uri)
        policy_architecture = artifact.policy_architecture
        embedded_policy = getattr(artifact, "policy", None)
        if embedded_policy is not None and class_path is None:
            embedded_policy_class_path = (
                f"{embedded_policy.__class__.__module__}.{embedded_policy.__class__.__qualname__}"
            )

    if policy_architecture is None and class_path is None and embedded_policy_class_path is None:
        raise ValueError("policy_architecture or class_path is required for checkpoints without embedded metadata")

    init_kwargs: dict[str, str | bool | PolicyArchitecture] = {
        "checkpoint_uri": normalized_uri,
        "display_name": display_name or normalized_uri,
        "strict": strict,
    }
    if device is not None:
        init_kwargs["device"] = str(device)

    resolved_class_path = (
        policy_architecture.class_path
        if policy_architecture is not None
        else (class_path or embedded_policy_class_path)
    )
    if resolved_class_path is None:
        raise ValueError("policy_architecture or class_path is required to build a PolicySpec")
    if policy_architecture is not None:
        init_kwargs["policy_architecture"] = policy_architecture

    data_path = (
        str(parsed_uri.local_path)
        if parsed_uri.scheme == "file" and parsed_uri.local_path and parsed_uri.local_path.is_file()
        else None
    )

    return PolicySpec(class_path=resolved_class_path, init_kwargs=init_kwargs, data_path=data_path)
