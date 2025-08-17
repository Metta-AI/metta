"""Common initialization for Metta scripts."""

import argparse
import importlib
import inspect
import json
import logging
import os
import signal
import sys
from types import FrameType
from typing import Callable, TypeVar, cast

from omegaconf import OmegaConf
from pydantic import BaseModel

from metta.common.util.logging_helpers import init_logging

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


def pydantic_metta_script(main: Callable[[T], int | None]) -> None:
    """
    Wrapper for Metta script entry points that performs environment setup and
    configuration before calling the `main` function.

    Example usage:
    ```python
    from metta.util.metta_script import metta_script

    class MyToolConfig(Config):
        ...

    def main(cfg: MyToolConfig):
        ...

    pydantic_metta_script(main)
    ```

    Calling this function will do nothing if the script is loaded as a module.

    The script can be run with:
    ```bash
    ./tools/my_tool.py --cfg configs/my_tool.yaml
    ./tools/my_tool.py --func metta.tools.my_tool.main
    ./tools/my_tool.py --func metta.tools.my_tool.main my.param.override=value
    ```

    This wrapper:
    1. Parses CLI arguments
    2. Initializes logging to both stdout and run_dir/logs/
    3. Initializes the runtime environment for MettaGrid simulations:
       - Sets up environment variables
       - Initializes random seeds
    """
    # If not running as a script, there's nothing to do.
    caller_frame: FrameType = inspect.stack()[1].frame
    caller_globals = caller_frame.f_globals
    if caller_globals.get("__name__") != "__main__":
        return

    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, required=False)
    parser.add_argument("--cfg", type=str, required=False)
    args, override_args = parser.parse_known_args()
    overrides_conf = OmegaConf.from_cli(override_args)

    init_logging()

    # Exit on ctrl+c
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

    # Detect the Pydantic model
    config_class = main.__annotations__.get("cfg")
    if config_class is None:
        raise ValueError("Main function must have a cfg parameter")
    if isinstance(config_class, str):
        raise ValueError(
            "cfg parameter must be a Pydantic model, got str, are you using `from __future__ import annotations`?"
        )
    if not issubclass(config_class, BaseModel):
        raise ValueError(f"cfg parameter must be a Pydantic model, got {config_class}")

    # Load the config and apply overrides
    if args.cfg:
        with open(args.cfg, "r") as f:
            conf = OmegaConf.merge(json.load(f), overrides_conf)
            cfg = config_class.model_validate(conf)
    elif args.func:
        module_name, func_name = args.func.rsplit(".", 1)
        cfg = importlib.import_module(module_name).__getattribute__(func_name)()
        cfg = config_class.model_validate(OmegaConf.to_container(OmegaConf.merge(cfg.model_dump(), overrides_conf)))
    else:
        cfg = config_class.model_validate(OmegaConf.to_container(overrides_conf))

    assert isinstance(cfg, config_class)
    cfg = cast(T, cfg)

    # Determine the filename where `main` is defined and print it relative to CWD
    main_source_file = inspect.getsourcefile(main) or inspect.getfile(main)
    if main_source_file:
        main_source_relpath = os.path.relpath(os.path.abspath(main_source_file), os.getcwd())
    else:
        main_source_relpath = f"{main.__module__}:<unknown>"

    logger.info(f"Running {main_source_relpath} with config:\n{cfg.model_dump_json(indent=2)}")

    result = main(cfg)
    if result is not None:
        sys.exit(result)
