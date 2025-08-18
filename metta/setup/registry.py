from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from metta.setup.components.base import SetupModule

_REGISTRY: list[Type["SetupModule"]] = []


def register_module(cls: Type["SetupModule"]) -> Type["SetupModule"]:
    _REGISTRY.append(cls)
    return cls


def get_all_modules(config) -> list["SetupModule"]:
    all_modules = [cls(config) for cls in _REGISTRY]

    # Create a mapping from name to module for easy lookup
    name_to_module = {m.name: m for m in all_modules}

    # Topological sort based on dependencies
    visited: set[str] = set()
    temp_visited: set[str] = set()
    result: list["SetupModule"] = []

    def visit(module: "SetupModule") -> None:
        if module.name in temp_visited:
            raise ValueError(f"Circular dependency detected involving setup module {module.name}")
        if module.name in visited:
            return

        temp_visited.add(module.name)

        # Visit all dependencies first
        for dep_name in module.dependencies():
            if dep_name in name_to_module:
                visit(name_to_module[dep_name])

        temp_visited.remove(module.name)
        visited.add(module.name)
        result.append(module)

    # Visit all modules
    for module in all_modules:
        if module.name not in visited:
            visit(module)

    return result


def get_applicable_modules(config) -> list["SetupModule"]:
    return [m for m in get_all_modules(config) if m.is_applicable()]
