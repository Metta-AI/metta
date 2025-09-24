import logging
import os
import socket
from typing import Any

import wandb
import wandb.sdk.wandb_run
from wandb.errors import CommError

from mettagrid.config import Config

logger = logging.getLogger(__name__)

# Alias type for easier usage (other modules can import this type)
WandbRun = wandb.sdk.wandb_run.Run


class WandbConfig(Config):
    enabled: bool
    project: str
    entity: str
    group: str | None = None
    name: str | None = None
    run_id: str | None = None
    data_dir: str | None = None
    job_type: str | None = None
    tags: list[str] = []
    notes: str = ""

    @staticmethod
    def Off() -> "WandbConfig":
        return WandbConfig(enabled=False, project="na", entity="na")

    # Has the same behavior as Off, but indicates that it should be replaced by wandb_auto_config
    @staticmethod
    def Unconfigured() -> "WandbConfig":
        return WandbConfig(enabled=False, project="unconfigured", entity="unconfigured")

    @property
    def uri(self):
        raise RuntimeError("Policy artifacts are no longer stored on WandB. Use local or s3:// URIs instead.")


class WandbContext:
    """
    Context manager for Wandb.

    Usually initialized in the following way:

        with WandbContext(wandb_cfg) as wandb_run:
            ...

    Or with extra configuration:

        with WandbContext(wandb_cfg, extra_cfg=config) as wandb_run:
            ...
    """

    def __init__(
        self,
        wandb_cfg: WandbConfig,
        extra_cfg: Config | dict[str, Any] | None = None,
        timeout: int = 30,
        extra_cfg_name: str | None = None,
    ):
        """
        Initialize WandbContext.

        Args:
            wandb_cfg: WandB configuration
            extra_cfg: Optional Config object or dict to log to the wandb run
            timeout: Connection timeout in seconds
        """
        self.cfg = wandb_cfg
        self.extra_cfg = extra_cfg
        self.extra_cfg_name = extra_cfg_name
        self.run: WandbRun | None = None
        self.timeout = timeout  # Add configurable timeout (wandb default is 90 seconds)
        self.wandb_host = "api.wandb.ai"
        self.wandb_port = 443
        self._generated_ipc_file_path: str | None = None  # To store path if generated

    def __enter__(self) -> WandbRun | None:
        if not self.cfg.enabled:
            return None

        assert self.cfg.enabled

        # Check internet connection before proceeding
        try:
            socket.setdefaulttimeout(5)  # Set a 5-second timeout for the connection check
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.wandb_host, self.wandb_port))
            logger.info(f"Connection to {self.wandb_host} verified")
        except Exception as ex:
            logger.warning(f"No connection to {self.wandb_host} servers detected: {str(ex)}")
            logger.info("Continuing without W&B logging")
            return None

        logger.info(f"Initializing W&B run with timeout={self.timeout}s")

        try:
            tags = list(self.cfg.tags)
            tags.append("user:" + os.environ.get("METTA_USER", "unknown"))

            # Build config dict
            config = None
            if self.extra_cfg:
                if isinstance(self.extra_cfg, dict):
                    key = self.extra_cfg_name or "extra_config_dict"
                    config = {key: self.extra_cfg}
                else:
                    # Assume it's a Config object with model_dump method
                    class_name = self.extra_cfg.__class__.__name__
                    key = self.extra_cfg_name or class_name or "extra_config_object"
                    config = {key: self.extra_cfg.model_dump()}

            self.run = wandb.init(
                id=self.cfg.run_id,
                job_type=self.cfg.job_type,
                project=self.cfg.project,
                entity=self.cfg.entity,
                config=config,
                group=self.cfg.group,
                allow_val_change=True,
                monitor_gym=True,
                save_code=True,
                resume=True,
                tags=tags,
                notes=self.cfg.notes or None,
                settings=wandb.Settings(quiet=True, init_timeout=self.timeout),
            )

            # Save config and set up file syncing only if wandb init succeeded and data_dir is set
            if self.cfg.data_dir:
                wandb.save(os.path.join(self.cfg.data_dir, "*.log"), base_path=self.cfg.data_dir, policy="live")
                wandb.save(os.path.join(self.cfg.data_dir, "*.yaml"), base_path=self.cfg.data_dir, policy="live")
                wandb.save(os.path.join(self.cfg.data_dir, "*.json"), base_path=self.cfg.data_dir, policy="live")
            logger.info(f"Successfully initialized W&B run: {self.run.name} ({self.run.id})")

        except (TimeoutError, CommError) as e:
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
