#!/usr/bin/env -S uv run
import argparse
import inspect
import logging
import os
import signal
import sys
import warnings
from typing import Any, cast

from omegaconf import OmegaConf
from typing_extensions import TypeVar

from metta.common.tool import Tool
from metta.common.util.log_config import init_logging
from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol
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


T = TypeVar("T", bound=Config)


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("make_tool_cfg_path", type=str, help="Path to the function to run")
    parser.add_argument("--args", nargs="*")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true", default=False)
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
    make_tool_cfg = load_symbol(args.make_tool_cfg_path)

    if issubclass(make_tool_cfg, Tool):
        # tool config constructor
        tool_cfg = make_tool_cfg(**args_conf)
    else:
        # function that makes a tool config
        make_tool_cfg_args = {}
        for key in inspect.signature(make_tool_cfg).parameters.keys():
            if key in args_conf:
                make_tool_cfg_args[key] = args_conf[key]
        tool_cfg = make_tool_cfg(**make_tool_cfg_args)
        used_args = set(make_tool_cfg_args.keys())
        used_args.update(tool_cfg.consumed_args)
        unused_args = set(args_conf.keys()) - used_args
        if unused_args:
            raise ValueError(f"Unused arguments passed to {args.make_tool_cfg_path}: {unused_args}")

    if not isinstance(tool_cfg, Tool):
        raise ValueError(f"The result of running {args.make_tool_cfg_path} must be a ToolConfig, got {tool_cfg}")

    overrides = args.overrides or []
    for override in overrides:
        key, value = override.split("=", 1)
        tool_cfg = tool_cfg.override(key, value)

    # Seed random number generators
    seed_everything(tool_cfg.system)

    # Run the tool from config
    if args.dry_run:
        print("Dry run: printing tool config")
        print(tool_cfg.model_dump_json(indent=2))
        sys.exit(0)

    result = tool_cfg.invoke(args_conf, overrides)

    if result is not None:
        sys.exit(result)
