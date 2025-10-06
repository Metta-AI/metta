from __future__ import annotations

import ast
import inspect
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, cast, get_args

from metta.common.util.fs import get_repo_root
from metta.gridworks.configs.lsp import LSPClient
from mettagrid.base_config import Config
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)

ConfigMakerKind = Literal[
    "MettaGridConfig",
    "SimulationConfig",
    "List[SimulationConfig]",
    "TrainTool",
    "PlayTool",
    "ReplayTool",
    "EvaluateTool",
    "CurriculumConfig",
]


def hover_value_to_return_type(hover_value: str) -> str:
    post_arrow = hover_value.split(" -> ")[1]
    match = re.match(r"^[\w\[\]]+", post_arrow)
    if match:
        return match.group(0)
    else:
        return "Unknown"


def check_return_type(return_type: str) -> ConfigMakerKind | None:
    # normalize
    return_type = re.sub(r"\blist\b", "List", return_type)
    if return_type in get_args(ConfigMakerKind):
        return cast(ConfigMakerKind, return_type)

    return None


MakerFunction = Callable[[], Config | list[Config]]


@dataclass
class ConfigMaker:
    """Represents a function that makes a Config object, and its metadata."""

    maker: MakerFunction
    return_type: ConfigMakerKind
    line: int

    def path(self) -> str:
        return self.maker.__module__ + "." + self.maker.__name__

    def to_dict(self) -> dict:
        return {
            "kind": self.return_type,
            "path": self.path(),
            "absolute_path": inspect.getfile(self.maker),
            "line": self.line,
        }

    @classmethod
    def from_path(cls, path: str, return_type: ConfigMakerKind, line: int) -> ConfigMaker:
        maker = load_symbol(path)
        if not callable(maker):
            raise ValueError(f"Symbol {path} is not a callable")

        sig = inspect.signature(maker)

        # Check if all parameters have defaults
        for param in sig.parameters.values():
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"Symbol {path} must have no required arguments (all parameters must have defaults)")

        return ConfigMaker(maker=cast(MakerFunction, maker), return_type=return_type, line=line)


class ConfigMakerRegistry:
    """Registry of all config makers."""

    def __init__(self, root_dir: Path | None = None):
        if not root_dir:
            repo_root = get_repo_root()
            root_dir = repo_root / "experiments"
        else:
            root_dir = root_dir.resolve()

        self.lsp_client = LSPClient()

        # Load all config makers
        # TODO - implement reloading (full and maybe incremental based on file modification time)
        config_makers: list[ConfigMaker] = []
        for py_file in root_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                file_config_makers = self.load_file_config_makers(py_file)
                config_makers.extend(file_config_makers)
            except Exception as e:
                logger.info(f"Error getting config makers from {py_file}: {e}")
                # Skip files that can't be parsed or processed
                continue

        self._config_makers = config_makers
        self._config_makers_index = {maker.path(): maker for maker in config_makers}

    def load_file_config_makers(self, file_path: Path) -> list[ConfigMaker]:
        code = file_path.read_text()
        tree = ast.parse(code)
        config_makers: list[ConfigMaker] = []
        function_defs: list[ast.FunctionDef] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Top-level function
                function_defs.append(node)

        logging.debug(f"Found {len(function_defs)} function definitions in {file_path}")

        # TODO - what if the code changes between `ast.parse` and `get_hover_bulk`?
        # Should we pre-open the file with LSP during loading?
        hover_results = self.lsp_client.get_hover_bulk(
            file_path,
            # TODO: this line detection code might not work well for decorated functions.
            # (It's surprisingly hard to get the function name location from the AST, because the name is not stored as
            # an individual node; if this becomes too fragile, we'll need either a third-party library, or regexes).
            [(node.lineno - 1, node.col_offset + 4) for node in function_defs],
        )
        logging.debug(f"Found {len(hover_results)} hover results in {file_path}")

        for node, hover_result in zip(function_defs, hover_results, strict=True):
            # Create full module path
            rel_file_path = file_path.relative_to(get_repo_root())
            module_path = str(rel_file_path.with_suffix("")).replace("/", ".")
            full_path = f"{module_path}.{node.name}"

            try:
                assert hover_result is not None
                return_type = hover_value_to_return_type(hover_result["contents"]["value"])

                validated_return_type = check_return_type(return_type)
                if not validated_return_type:
                    continue

                config_maker = ConfigMaker.from_path(full_path, validated_return_type, node.lineno)
                config_makers.append(config_maker)
            except Exception as e:
                logger.info(f"Error getting config maker from {full_path}: {e}")
                # Skip if can't load or doesn't meet requirements
                pass
        return config_makers

    def size(self) -> int:
        return len(self._config_makers)

    def grouped_by_kind(self) -> dict[ConfigMakerKind, list[ConfigMaker]]:
        grouped: dict[ConfigMakerKind, list[ConfigMaker]] = {}
        for maker in self._config_makers:
            if maker.return_type not in grouped:
                grouped[maker.return_type] = []
            grouped[maker.return_type].append(maker)
        return grouped

    def get_by_path(self, path: str) -> ConfigMaker | None:
        return self._config_makers_index.get(path)
