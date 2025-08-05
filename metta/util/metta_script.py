"""Common initialization for Metta scripts."""

import functools
import inspect
import logging
import os
import platform
import signal
import sys
from types import FrameType
from typing import Callable

import hydra
from omegaconf import DictConfig, ListConfig

from metta.common.util.fs import get_repo_root
from metta.common.util.logging_helpers import init_logging
from metta.common.util.resolvers import register_resolvers
from metta.util.init.mettagrid_environment import init_mettagrid_environment

logger = logging.getLogger(__name__)


def apply_mac_device_overrides(cfg: DictConfig) -> None:
    if not cfg.bypass_mac_overrides and platform.system() == "Darwin":
        cfg.device = "cpu"
        cfg.vectorization = "serial"


def metta_script(
    main: Callable[[DictConfig], int | None],
    config_name: str,
    pre_main: Callable[[DictConfig], None] | None = None,
) -> None:
    """
    Wrapper for Metta script entry points that performs environment setup and
    configuration before calling the `main` function.

    Example usage:
    ```python
    from metta.util.metta_script import metta_script

    def main(cfg: DictConfig):
        ...

    # call main() with the config from configs/my_job.yaml
    metta_script(main, "my_job")
    ```

    Calling this function will do nothing if the script is loaded as a module.

    This wrapper:
    1. Configures Hydra to load the `config_name` config and pass it to the `main` function
    2. Applies device overrides for Mac
    3. Calls the optional `pre_main` if provided
    4. Initializes logging to both stdout and run_dir/logs/
    5. Initializes the runtime environment for MettaGrid simulations:
       - Create required directories (including run_dir)
       - Configure CUDA settings
       - Set up environment variables
       - Initialize random seeds
       - Register OmegaConf resolvers
    6. Performs device validation and sets the device to "cpu" if CUDA is not available
    """

    # If not running as a script, there's nothing to do.
    caller_frame: FrameType = inspect.stack()[1].frame
    caller_globals = caller_frame.f_globals
    if caller_globals.get("__name__") != "__main__":
        return

    script_path = caller_globals["__file__"]

    # Wrapped main function that we want to run.
    # This code runs after the Hydra was configured. Depending on CLI args such as `--help`, it may not run at all.
    def extended_main(cfg: ListConfig | DictConfig) -> None:
        if not isinstance(cfg, DictConfig):
            raise ValueError("Metta scripts must be run with a DictConfig")

        if pre_main:
            pre_main(cfg)

        if cfg.agent_path:
            cfg.agent = cfg.agent_path

        apply_mac_device_overrides(cfg)

        run_dir = cfg.get("run_dir")
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

        # Initialize logging
        init_logging(run_dir=run_dir)

        logger.info(f"Starting {main.__name__} from {script_path} with run_dir: {run_dir or 'not set'}")

        # Initialize the full mettagrid environment (includes device validation)
        init_mettagrid_environment(cfg)

        logger.info("Environment setup completed")

        # Exit on ctrl+c
        signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

        # Call the original function
        result = main(cfg)
        if result is not None:
            sys.exit(result)

    # Hydra analyzes the wrapped function, and the function must come from the
    # `__main__` name for hydra to work correctly.
    # So we have to pretend that we wrap the original function from the script,
    # not the `extended_main()` function defined above.
    functools.update_wrapper(extended_main, main)

    # Hydra needs the config path to be relative to the original script.
    script_dir = os.path.abspath(os.path.dirname(script_path))
    abs_config_path = str(get_repo_root() / "configs")
    relative_config_path = os.path.relpath(abs_config_path, script_dir)

    # Calling `hydra.main` as a function instead of a decorator, because `extended_main` function
    # needs to be patched with `functools.update_wrapper` first.
    configured_main = hydra.main(config_path=relative_config_path, config_name=config_name, version_base=None)(
        extended_main
    )

    configured_main()


def hydraless_metta_script(main: Callable[[], int | None]) -> None:
    """
    Wrapper for Metta scripts that does not use Hydra.
    """
    # If not running as a script, there's nothing to do.
    caller_frame: FrameType = inspect.stack()[1].frame
    caller_globals = caller_frame.f_globals
    if caller_globals.get("__name__") != "__main__":
        return

    init_logging()
    register_resolvers()

    # Exit on ctrl+c
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

    # Call the original function
    result = main()
    if result is not None:
        sys.exit(result)
