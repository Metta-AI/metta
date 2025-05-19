import copy
import logging
import os
import socket
from typing import Annotated, Literal, Union, cast

import pkg_resources
import requests
import wandb
import wandb.errors
import wandb.sdk.wandb_run
import wandb.util
from omegaconf import OmegaConf
from pydantic import Field, TypeAdapter

from metta.util.config import Config

logger = logging.getLogger(__name__)

# Alias type for easier usage (other modules can import this type)
WandbRun = wandb.sdk.wandb_run.Run


class WandbConfigOn(Config):
    enabled: Literal[True] = True

    project: str
    entity: str
    group: str
    name: str
    run_id: str
    data_dir: str
    job_type: str


class WandbConfigOff(Config, extra="allow"):
    enabled: Literal[False] = False


WandbConfig = Annotated[Union[WandbConfigOff, WandbConfigOn], Field(discriminator="enabled")]


def check_wandb_version() -> bool:
    try:
        # Get the installed wandb version
        installed_version = pkg_resources.get_distribution("wandb").version
        installed_minor = int(installed_version.split(".")[1])

        # Fetch latest version from GitHub API
        response = requests.get("https://api.github.com/repos/wandb/wandb/releases/latest")
        if response.status_code == 200:
            latest_version = response.json()["tag_name"].lstrip("v")
            required_minor = int(latest_version.split(".")[1])
            logger.info(f"wandb installed version is {installed_version}, latest version is {latest_version}")

            # Check if the installed version meets the required minor version
            if installed_minor < required_minor:
                logger.error(f"Your wandb version ({installed_version}) is outdated.")
                logger.error(f"Required version is 0.{required_minor}.x or later based on latest GitHub release.")
                logger.error(f"Latest available version is {latest_version}")
                logger.error("Please update using: pip install --upgrade wandb")
                return False
            else:
                return True

    except pkg_resources.DistributionNotFound:
        logger.error("wandb package is not installed.")
        logger.error("Please install wandb using: pip install wandb")
        return False

    except requests.RequestException as e:
        logger.warning(f"Could not check latest version for wandb package from GitHub: {e}")

    return True


class WandbContext:
    """
    Context manager for Wandb.

    Usually initialized in the following way:

        with WandbContext(cfg.wandb, cfg) as wandb_run:
            ...
    """

    def __init__(
        self,
        # Either a `DictConfig` from Hydra, or already validated `WandbConfig` object.
        cfg: object,
        # Global Hydra config, needed because we store it to WanDB.
        global_cfg: object,
        timeout: int = 30,
    ):
        if isinstance(cfg, (WandbConfigOn, WandbConfigOff)):
            self.cfg = cfg
        else:
            # validate
            self.cfg = TypeAdapter(WandbConfig).validate_python(cfg)

        self.global_cfg = global_cfg

        self.run = None
        self.timeout = timeout  # Add configurable timeout (wandb default is 90 seconds)
        self.wandb_host = "api.wandb.ai"
        self.wandb_port = 443

        check_wandb_version()

    def __enter__(self) -> WandbRun | None:
        if not self.cfg.enabled:
            return None

        assert isinstance(self.cfg, WandbConfigOn)

        # Check internet connection before proceeding
        try:
            socket.setdefaulttimeout(5)  # Set a 5-second timeout for the connection check
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.wandb_host, self.wandb_port))
            logger.info(f"Connection to {self.wandb_host} verified")
        except Exception as ex:
            logger.warning(f"No connection to {self.wandb_host} servers detected: {str(ex)}")
            logger.info("Continuing without W&B logging")
            return None

        global_cfg = copy.deepcopy(self.global_cfg)
        logger.info(f"Initializing W&B run with timeout={self.timeout}s")

        try:
            self.run = wandb.init(
                id=self.cfg.run_id,
                job_type=self.cfg.job_type,
                project=self.cfg.project,
                entity=self.cfg.entity,
                config=cast(dict, OmegaConf.to_container(global_cfg, resolve=False)),
                group=self.cfg.group,
                allow_val_change=True,
                name=self.cfg.name,
                monitor_gym=True,
                save_code=True,
                resume=True,
                tags=["user:" + os.environ.get("METTA_USER", "unknown")],
                settings=wandb.Settings(quiet=True, init_timeout=self.timeout),
            )

            # Save config and set up file syncing only if wandb init succeeded
            OmegaConf.save(global_cfg, os.path.join(self.cfg.data_dir, "config.yaml"))
            wandb.save(os.path.join(self.cfg.data_dir, "*.log"), base_path=self.cfg.data_dir, policy="live")
            wandb.save(os.path.join(self.cfg.data_dir, "*.yaml"), base_path=self.cfg.data_dir, policy="live")
            logger.info(f"Successfully initialized W&B run: {self.run.name} ({self.run.id})")

        except (TimeoutError, wandb.errors.CommError) as e:
            error_type = "timeout" if isinstance(e, TimeoutError) else "communication"
            logger.warning(f"W&B initialization failed due to {error_type} error: {str(e)}")
            logger.info("Continuing without W&B logging")
            self.run = None

        except Exception as e:
            logger.error(f"Unexpected error during W&B initialization: {str(e)}")
            logger.info("Continuing without W&B logging")
            self.run = None

        return self.run

    @staticmethod
    def cleanup_run(run: WandbRun | None):
        if run:
            try:
                wandb.finish()
            except Exception as e:
                logger.error(f"Error during W&B cleanup: {str(e)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_run(self.run)
