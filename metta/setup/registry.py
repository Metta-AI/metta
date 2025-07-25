from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from metta.setup.components.base import SetupModule

_REGISTRY: list[Type["SetupModule"]] = []


def register_module(cls: Type["SetupModule"]) -> Type["SetupModule"]:
    _REGISTRY.append(cls)
    return cls


def get_all_modules(config) -> list["SetupModule"]:
    all_modules = [cls(config) for cls in _REGISTRY]
    # Sort by name for consistent ordering
    all_modules.sort(key=lambda m: m.name)

    return all_modules


def get_applicable_modules(config) -> list["SetupModule"]:
    return [m for m in get_all_modules(config) if m.is_applicable()]
