from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Callable, Literal, cast

from metta.common.util.fs import get_repo_root
from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol

ConfigMakerKind = Literal["MettaGrid", "Simulation", "List[Simulation]"]


class ConfigMaker:
    """Represents a function that makes a config."""

    def __init__(self, maker: Callable[[], Config]):
        self._maker = maker

    def kind(self) -> ConfigMakerKind:
        return "MettaGrid"  # TODO

    def to_dict(self) -> dict:
        return {
            "kind": self.kind(),
            "path": self._maker.__module__ + "." + self._maker.__name__,
            "absolute_path": inspect.getfile(self._maker),
        }

    @classmethod
    def from_path(cls, path: str) -> ConfigMaker:
        maker = load_symbol(path)
        if not callable(maker):
            raise ValueError(f"Symbol {path} is not a callable")

        sig = inspect.signature(maker)

        # Check if all parameters have defaults
        for param in sig.parameters.values():
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"Symbol {path} must have no required arguments (all parameters must have defaults)")

        if not issubclass(sig.return_annotation, Config):
            raise ValueError(f"Symbol {path} must return a Config")

        return ConfigMaker(cast(Callable[[], Config], maker))


class ConfigMakerRegistry:
    """Registry of all config makers."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        repo_root = get_repo_root()
        experiments_dir = repo_root / "experiments"

        config_makers: list[ConfigMaker] = []

        # Find all Python files in experiments/
        for py_file in experiments_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            # Get config makers from this file
            try:
                file_config_makers = self._get_config_makers(py_file)
                config_makers.extend(file_config_makers)
            except Exception as e:
                print(f"Error getting config makers from {py_file}: {e}")
                # Skip files that can't be parsed or processed
                continue

        self._config_makers = config_makers

    def _get_config_makers(self, file_path: Path) -> list[ConfigMaker]:
        with open(file_path, "r") as f:
            code = f.read()
        tree = ast.parse(code)
        visitors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.col_offset == 0:  # Top-level function
                # Check if it has a return annotation of MettaGridConfig
                if node.returns and isinstance(node.returns, ast.Name) and node.returns.id == "MettaGridConfig":
                    print(file_path, node.name)
                    # Create full module path
                    rel_file_path = file_path.relative_to(get_repo_root())
                    module_path = str(rel_file_path.with_suffix("")).replace("/", ".")
                    full_path = f"{module_path}.{node.name}"
                    try:
                        config_maker = ConfigMaker.from_path(full_path)
                        visitors.append(config_maker)
                    except Exception as e:
                        print(f"Error getting config maker from {full_path}: {e}")
                        # Skip if can't load or doesn't meet requirements
                        pass
        return visitors

    def grouped_by_kind(self) -> dict[ConfigMakerKind, list[ConfigMaker]]:
        return {
            maker.kind(): [maker for maker in self._config_makers if maker.kind() == maker.kind()]
            for maker in self._config_makers
        }
