#!/usr/bin/env -S uv run
import argparse
import importlib
import logging
import os
import signal
import sys
import warnings
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import validate_call

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

    overrides_conf = OmegaConf.to_container(OmegaConf.from_cli(args.overrides or []))
    assert isinstance(overrides_conf, dict)
    overrides_conf = cast(dict[str, Any], overrides_conf)

    args_conf = OmegaConf.to_container(OmegaConf.from_cli(args.args or []))
    assert isinstance(args_conf, dict)
    args_conf = cast(dict[str, Any], args_conf)

    # Create the tool config object
    module_name, func_name = args.make_tool_cfg_path.rsplit(".", 1)
    make_tool_cfg = importlib.import_module(module_name).__getattribute__(func_name)

    tool_cfg = validate_call(make_tool_cfg)(**args_conf)
    if not isinstance(tool_cfg, Tool):
        raise ValueError(f"The result of running {args.make_tool_cfg_path} must be a ToolConfig, got {tool_cfg}")

    # Seed random number generators
    seed_everything(tool_cfg.system)

    # Run the tool from config
    logger.info(f"Running tool config: {module_name}.{func_name}")
    result = tool_cfg.invoke()

    if result is not None:
        sys.exit(result)


if __name__ == "__main__":
    main()
