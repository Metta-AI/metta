from __future__ import annotations

import ast
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.file import local_copy
from mettagrid.util.module import load_symbol


class PolicyArchitectureProtocol(Protocol):
    def make_policy(self, policy_env_info: PolicyEnvInterface) -> Any: ...
    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


def _component_config_to_manifest(component: Any) -> dict[str, Any]:
    data = component.model_dump(mode="json")
    data["class_path"] = f"{component.__class__.__module__}.{component.__class__.__qualname__}"
    return data


def _load_component_config(
    data: Any,
    *,
    context: str,
    default_class: type | None = None,
) -> Any:
    if not isinstance(data, Mapping):
        if hasattr(data, "model_dump"):
            return data
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


def architecture_to_string(architecture: Any) -> str:
    class_path = f"{architecture.__class__.__module__}.{architecture.__class__.__qualname__}"
    config_data = architecture.model_dump(mode="json")
    config_data.pop("class_path", None)

    if "components" in config_data:
        config_data["components"] = [_component_config_to_manifest(c) for c in architecture.components]

    action_probs_config = getattr(architecture, "action_probs_config", None)
    if action_probs_config is not None:
        config_data["action_probs_config"] = _component_config_to_manifest(action_probs_config)

    if not config_data:
        return class_path

    sorted_config = _sorted_structure(config_data)
    parts = [f"{key}={repr(sorted_config[key])}" for key in sorted(sorted_config)]
    args_repr = ", ".join(parts)
    return f"{class_path}({args_repr})"


def architecture_from_string(spec: str) -> Any:
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
    if not isinstance(config_class, type):
        raise TypeError(f"Loaded symbol {class_path} is not a class")

    payload: dict[str, Any] = dict(kwargs)

    default_components: list[Any] = []
    default_action_probs: Any = None
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

    return config_class.model_validate(payload)


def _to_safetensors_state_dict(
    state_dict: Mapping[str, torch.Tensor], detach_buffers: bool
) -> MutableMapping[str, torch.Tensor]:
    ordered: MutableMapping[str, torch.Tensor] = OrderedDict()
    seen_storage: dict[int, str] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State dict entry '{key}' is not a torch.Tensor")
        value = tensor.detach() if detach_buffers else tensor
        value_cpu = value.cpu()
        data_ptr = value_cpu.data_ptr()
        if data_ptr in seen_storage:
            value_cpu = value_cpu.clone()
        else:
            seen_storage[data_ptr] = key
        ordered[key] = value_cpu
    return ordered


@dataclass
class MptArtifact:
    architecture: Any
    state_dict: MutableMapping[str, torch.Tensor]

    def instantiate(
        self,
        policy_env_info: PolicyEnvInterface,
        device: torch.device | str = "cpu",
        *,
        strict: bool = True,
    ) -> Any:
        if isinstance(device, str):
            device = torch.device(device)

        policy = self.architecture.make_policy(policy_env_info)
        policy = policy.to(device)

        if hasattr(policy, "initialize_to_environment"):
            policy.initialize_to_environment(policy_env_info, device)

        ordered_state = OrderedDict(self.state_dict.items())
        missing, unexpected = policy.load_state_dict(ordered_state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

        return policy


DEFAULT_URI_RESOLVER = "mettagrid.util.url_schemes.resolve_uri"


def load_mpt(uri: str, *, uri_resolver: str | None = DEFAULT_URI_RESOLVER) -> MptArtifact:
    """Load an .mpt checkpoint from a URI (file://, s3://, or local path).

    Args:
        uri: The URI to load from. Supports file://, s3://, and local paths.
             If a uri_resolver is configured, also supports metta:// and :latest URIs.
        uri_resolver: Optional dotted path to a URI resolver function. The function
            should accept a URI string and return a resolved URI string.
            Defaults to mettagrid.util.url_schemes.resolve_uri.
    """
    resolved_uri = uri
    if uri_resolver and (resolver_func := load_symbol(uri_resolver, strict=False)):
        resolved_uri = resolver_func(uri)  # type: ignore

    with local_copy(resolved_uri) as local_path:
        return _load_local_mpt_file(local_path)


def _load_local_mpt_file(path: Path) -> MptArtifact:
    if not path.exists():
        raise FileNotFoundError(f"MPT file not found: {path}")

    with zipfile.ZipFile(path, mode="r") as archive:
        names = set(archive.namelist())

        if "modelarchitecture.txt" not in names or "weights.safetensors" not in names:
            raise ValueError(f"Invalid .mpt file: {path} (missing architecture or weights)")

        architecture_blob = archive.read("modelarchitecture.txt").decode("utf-8")
        architecture = architecture_from_string(architecture_blob)

        weights_blob = archive.read("weights.safetensors")
        state_dict = load_safetensors(weights_blob)
        if not isinstance(state_dict, MutableMapping):
            raise TypeError("Loaded safetensors state_dict is not a mutable mapping")

    return MptArtifact(architecture=architecture, state_dict=state_dict)


def save_mpt(
    uri: str | Path,
    *,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
    detach_buffers: bool = True,
) -> None:
    """Save an .mpt checkpoint to a URI or local path."""
    from mettagrid.util.file import ParsedURI, write_file

    parsed = ParsedURI.parse(str(uri))

    if parsed.scheme == "s3":
        with tempfile.NamedTemporaryFile(suffix=".mpt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _save_mpt_file_locally(
                tmp_path, architecture=architecture, state_dict=state_dict, detach_buffers=detach_buffers
            )
            write_file(parsed.canonical, str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        output_path = parsed.local_path or Path(str(uri)).expanduser().resolve()
        _save_mpt_file_locally(
            output_path, architecture=architecture, state_dict=state_dict, detach_buffers=detach_buffers
        )


def _save_mpt_file_locally(
    path: Path,
    *,
    architecture: Any,
    state_dict: Mapping[str, torch.Tensor],
    detach_buffers: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_state = _to_safetensors_state_dict(state_dict, detach_buffers)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

        try:
            with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                weights_blob = save_safetensors(artifact_state)
                archive.writestr("weights.safetensors", weights_blob)
                archive.writestr("modelarchitecture.txt", architecture_to_string(architecture))

            temp_path.replace(path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
