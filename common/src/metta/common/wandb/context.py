import logging
import os
import socket
from typing import TYPE_CHECKING, Any

from mettagrid.config import Config

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandbRun
else:
    WandbRun = Any

logger = logging.getLogger(__name__)


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
        wandb_config: WandbConfig,
        run_config: Config | dict[str, Any] | str | None = None,
        timeout: int = 30,
        run_config_name: str | None = None,
    ):
        """
        Initialize WandbContext.

        Args:
            wandb_cfg: WandB configuration
            run_config: Optional configuration data to log to the W&B run's config
            timeout: Connection timeout in seconds
            run_config_name: Optional name for the config when it's an object/dict

        """
        self.wandb_config = wandb_config
        self.run_config = run_config
        self.run_config_name = run_config_name
        self.run: WandbRun | None = None
        self.timeout = timeout  # Add configurable timeout (wandb default is 90 seconds)
        self.wandb_host = "api.wandb.ai"
        self.wandb_port = 443
        self._generated_ipc_file_path: str | None = None  # To store path if generated

    def __enter__(self) -> WandbRun | None:
        if not self.wandb_config.enabled:
            return None

        # Check for a live connection to W&B before proceeding
        try:
            socket.setdefaulttimeout(5)  # Set a 5-second timeout for the connection check
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.wandb_host, self.wandb_port))
            logger.info(f"Connection to {self.wandb_host} verified")
        except Exception as ex:
            logger.warning(f"No connection to {self.wandb_host} servers detected: {str(ex)}")
            logger.info("Continuing without W&B logging")
            return None

        logger.info(f"Initializing W&B run with timeout={self.timeout}s")

        import wandb
        from wandb.errors import CommError

        try:
            tags = list(self.wandb_config.tags)
            tags.append("user:" + os.environ.get("METTA_USER", os.environ.get("USER", "unknown")))

            # Build config dict
            config = None
            if self.run_config:
                if isinstance(self.run_config, dict):
                    key = self.run_config_name or "extra_config_dict"
                    config = {key: self.run_config}
                elif isinstance(self.run_config, Config):
                    # Assume it's a Config object with model_dump method
                    class_name = self.run_config.__class__.__name__
                    key = self.run_config_name or class_name or "extra_config_object"
                    config = {key: self.run_config.model_dump()}
                elif isinstance(self.run_config, str):
                    config = self.run_config
                else:
                    logger.error(f"Invalid extra_cfg: {self.run_config}")
                    config = None

            self.run = wandb.init(
                id=self.wandb_config.run_id,
                job_type=self.wandb_config.job_type,
                project=self.wandb_config.project,
                entity=self.wandb_config.entity,
                config=config,
                group=self.wandb_config.group,
                allow_val_change=True,
                monitor_gym=True,
                save_code=True,
                resume="allow",
                tags=tags,
                notes=self.wandb_config.notes or None,
                settings=wandb.Settings(quiet=True, init_timeout=self.timeout),
            )

            # Save config and set up file syncing only if wandb init succeeded and data_dir is set
            if self.wandb_config.data_dir:
                wandb.save(
                    os.path.join(self.wandb_config.data_dir, "*.log"),
                    base_path=self.wandb_config.data_dir,
                    policy="live",
                )
                wandb.save(
                    os.path.join(self.wandb_config.data_dir, "*.yaml"),
                    base_path=self.wandb_config.data_dir,
                    policy="live",
                )
                wandb.save(
                    os.path.join(self.wandb_config.data_dir, "*.json"),
                    base_path=self.wandb_config.data_dir,
                    policy="live",
                )
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
            import wandb

            try:
                wandb.finish()
            except Exception as e:
                logger.error(f"Error during W&B cleanup: {str(e)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_run(self.run)
