#!/usr/bin/env -S uv run
import argparse
import importlib
import logging
import os
import signal
import sys
import warnings
from typing import Any, NoReturn, cast

from omegaconf import OmegaConf
from pydantic import TypeAdapter, validate_call
from typing_extensions import TypeVar

from metta.common.util.config import Config
from metta.common.util.logging_helpers import init_logging
from metta.common.util.tool import Tool
from metta.rl.system_config import seed_everything

logger = logging.getLogger(__name__)


def init_mettagrid_system_environment() -> None:
    # Set CUDA launch blocking for better error messages in development
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Set environment variables to run without display
    os.environ["GLFW_PLATFORM"] = "osmesa"  # Use OSMesa as the GLFW backend
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["DISPLAY"] = ""

    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")


T = TypeVar("T", bound=Tool)


def apply_override(tool: T, key: str, value: Any) -> T:
    key_path = key.split(".")

    def fail(error: str) -> NoReturn:
        raise ValueError(f"Override failed. Full config:\n {tool.yaml()}\nOverride {key} failed: {error}")

    cfg: Config = tool
    traversed_path: list[str] = []
    for key_part in key_path[:-1]:
        if not hasattr(cfg, key_part):
            failed_path = ".".join(traversed_path + [key_part])
            fail(f"key {failed_path} not found")

        sub_cfg = getattr(cfg, key_part)
        if not isinstance(sub_cfg, Config):
            failed_path = ".".join(traversed_path + [key_part])
            fail(f"key {failed_path} is not a Config object")

        cfg = sub_cfg
        traversed_path.append(key_part)

    if not hasattr(cfg, key_path[-1]):
        fail(f"key {key} not found")

    cls = type(cfg)
    field = cls.model_fields.get(key_path[-1])
    if field is None:
        fail(f"key {key} is not a valid field")

    value = TypeAdapter(field.annotation).validate_python(value)
    setattr(cfg, key_path[-1], value)

    return tool


def apply_overrides(tool: T, overrides: list[str]) -> T:
    for override in overrides:
        key, value = override.split("=")
        apply_override(tool, key, value)

    return tool


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("make_tool_cfg_path", type=str, help="Path to the function to run")
    parser.add_argument("--args", nargs="*")
    parser.add_argument("--overrides", nargs="*")
    args = parser.parse_args()

    init_logging()
    init_mettagrid_system_environment()

    # Exit on ctrl+c
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

    make_tool_args = args.args or []
    assert isinstance(make_tool_args, list)

    args_conf = OmegaConf.to_container(OmegaConf.from_cli(make_tool_args))
    assert isinstance(args_conf, dict)
    args_conf = cast(dict[str, Any], args_conf)

    # Create the tool config object
    module_name, func_name = args.make_tool_cfg_path.rsplit(".", 1)
    make_tool_cfg = importlib.import_module(module_name).__getattribute__(func_name)

    tool_cfg = validate_call(make_tool_cfg)(**args_conf)
    if not isinstance(tool_cfg, Tool):
        raise ValueError(f"The result of running {args.make_tool_cfg_path} must be a ToolConfig, got {tool_cfg}")

    overrides = args.overrides or []
    tool_cfg = apply_overrides(tool_cfg, overrides)

    logger.info(
        f"Tool config produced by {args.make_tool_cfg_path}({', '.join(make_tool_args)}), "
        + f"with overrides {', '.join(overrides)}:"
        + "\n---------------------\n"
        + str(tool_cfg.yaml())
    )

    # Seed random number generators
    seed_everything(tool_cfg.system)

    # Run the tool from config
    result = tool_cfg.invoke()

    if result is not None:
        sys.exit(result)


if __name__ == "__main__":
    main()
