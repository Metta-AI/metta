from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, List

from pydantic import ConfigDict

from metta.agent.components.component_config import ComponentConfig
from mettagrid.base_config import Config
from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    from metta.agent.policy import Policy
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class PolicyArchitecture(Config):
    """Policy architecture configuration."""

    class_path: str

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    components: List[ComponentConfig] = []

    # a separate component that optionally accepts actions and process logits into log probs, entropy, etc.
    action_probs_config: ComponentConfig

    def make_policy(self, policy_env_info: PolicyEnvInterface) -> Policy:
        """Create an agent instance from configuration."""
        AgentClass = load_symbol(self.class_path)
        return AgentClass(policy_env_info, self)  # type: ignore[misc]

    def to_spec(self) -> str:
        """Serialize this architecture to a string specification."""
        class_path = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        config_data = self.model_dump(mode="json")
        config_data.pop("class_path", None)

        if "components" in config_data:
            config_data["components"] = [_component_to_manifest(c) for c in self.components]

        if self.action_probs_config is not None:
            config_data["action_probs_config"] = _component_to_manifest(self.action_probs_config)

        if not config_data:
            return class_path

        sorted_config = _sorted_structure(config_data)
        parts = [f"{key}={repr(sorted_config[key])}" for key in sorted(sorted_config)]
        return f"{class_path}({', '.join(parts)})"

    @classmethod
    def from_spec(cls, spec: str) -> PolicyArchitecture:
        """Deserialize an architecture from a string specification."""
        import ast

        spec = spec.strip()
        if not spec:
            raise ValueError("Policy architecture specification cannot be empty")

        expr = ast.parse(spec, mode="eval").body

        if isinstance(expr, ast.Call):
            class_path = _expr_to_dotted(expr.func)
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords if kw.arg}
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

        if "components" in payload:
            payload["components"] = [
                _load_component(
                    c, f"component[{i}]", default_components[i].__class__ if i < len(default_components) else None
                )
                for i, c in enumerate(payload["components"])
            ]

        if "action_probs_config" in payload:
            default_class = default_action_probs.__class__ if default_action_probs else None
            payload["action_probs_config"] = _load_component(
                payload["action_probs_config"], "action_probs_config", default_class
            )

        return config_class.model_validate(payload)


def _component_to_manifest(component: Any) -> dict[str, Any]:
    """Convert a component config to a serializable manifest with class_path."""
    data = component.model_dump(mode="json")
    data["class_path"] = f"{component.__class__.__module__}.{component.__class__.__qualname__}"
    return data


def _load_component(data: Any, context: str, default_class: type | None = None) -> Any:
    """Load a component config from serialized data."""
    from collections.abc import Mapping

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

    return component_class.model_validate(payload)


def _sorted_structure(value: Any) -> Any:
    """Recursively sort dicts by key for deterministic serialization."""
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {key: _sorted_structure(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_structure(item) for item in value]
    return value


def _expr_to_dotted(expr) -> str:
    """Convert an AST expression to a dotted class path string."""
    import ast

    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return f"{_expr_to_dotted(expr.value)}.{expr.attr}"
    raise ValueError("Expected a dotted name for policy architecture class path")
