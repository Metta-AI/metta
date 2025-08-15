#!/usr/bin/env -S uv run
import argparse
import importlib
import logging
import os
import signal
import sys
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import validate_call

from metta.common.util.logging_helpers import init_logging
from metta.common.util.tool import Tool

logger = logging.getLogger(__name__)


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("make_tool_cfg_path", type=str, help="Path to the function to run")
    parser.add_argument("--args", nargs="*")
    parser.add_argument("--overrides", nargs="*")
    args = parser.parse_args()

    init_logging()

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

    # Run the tool from config
    logger.info(f"Running tool config:\n{tool_cfg.yaml()}")
    result = tool_cfg.invoke()

    if result is not None:
        sys.exit(result)


if __name__ == "__main__":
    main()
