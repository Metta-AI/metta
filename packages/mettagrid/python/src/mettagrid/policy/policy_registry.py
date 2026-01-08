"""Policy registry system for automatic registration of classes with short_names."""

from __future__ import annotations

from abc import ABCMeta

# Registry mapping policy short names to full class paths
_POLICY_REGISTRY: dict[str, str] = {}


class PolicyRegistryMeta(type):
    """Metaclass that automatically registers classes with their short_names."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)

        # Get short_names from class namespace
        short_names = namespace.get("short_names")

        # Register each name with the full class path
        if short_names:
            module = namespace.get("__module__", "")
            full_class_path = f"{module}.{name}"
            for name_to_register in short_names:
                if name_to_register in _POLICY_REGISTRY:
                    existing_path = _POLICY_REGISTRY[name_to_register]
                    if existing_path != full_class_path:
                        msg = (
                            f"Policy short name '{name_to_register}' is already registered "
                            f"to {existing_path}, cannot register {full_class_path}"
                        )
                        raise ValueError(msg)
                _POLICY_REGISTRY[name_to_register] = full_class_path

        return cls


def get_policy_registry() -> dict[str, str]:
    """Get the policy registry mapping short names to full class paths."""
    return _POLICY_REGISTRY.copy()


class PolicyRegistryABCMeta(PolicyRegistryMeta, ABCMeta):
    """Combined metaclass for classes that need both PolicyRegistryMeta and ABCMeta."""

    pass
