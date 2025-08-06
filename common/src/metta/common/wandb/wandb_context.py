import copy
import json
import logging
import os
import socket
from typing import Annotated, Literal, Union, cast

import wandb
import wandb.errors
import wandb.sdk.wandb_run
from omegaconf import OmegaConf
from pydantic import Field, TypeAdapter

from metta.common.util.config import Config

logger = logging.getLogger(__name__)

# Alias type for easier usage (other modules can import this type)
WandbRun = wandb.sdk.wandb_run.Run
# Shared IPC filename, co-located with the heartbeat signal file
WANDB_IPC_FILENAME = "wandb_ipc.json"


class WandbConfigOn(Config):
    enabled: Literal[True] = True

    project: str
    entity: str
    group: str
    name: str
    run_id: str
    data_dir: str
    job_type: str
    tags: list[str] = []
    notes: str = ""


class WandbConfigOff(Config, extra="allow"):
    enabled: Literal[False] = False


WandbConfig = Annotated[Union[WandbConfigOff, WandbConfigOn], Field(discriminator="enabled")]


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

        self.run: WandbRun | None = None
        self.timeout = timeout  # Add configurable timeout (wandb default is 90 seconds)
        self.wandb_host = "api.wandb.ai"
        self.wandb_port = 443
        self._generated_ipc_file_path: str | None = None  # To store path if generated

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
            tags = list(self.cfg.tags)
            tags.append("user:" + os.environ.get("METTA_USER", "unknown"))
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
                tags=tags,
                notes=self.cfg.notes or None,
                settings=wandb.Settings(quiet=True, init_timeout=self.timeout),
            )

            # Save config and set up file syncing only if wandb init succeeded
            OmegaConf.save(global_cfg, os.path.join(self.cfg.data_dir, "config.yaml"))
            wandb.save(os.path.join(self.cfg.data_dir, "*.log"), base_path=self.cfg.data_dir, policy="live")
            wandb.save(os.path.join(self.cfg.data_dir, "*.yaml"), base_path=self.cfg.data_dir, policy="live")
            logger.info(f"Successfully initialized W&B run: {self.run.name} ({self.run.id})")

            # --- File-based IPC: Write to the same directory as HEARTBEAT_FILE ---
            heartbeat_file_env_path = os.environ.get("HEARTBEAT_FILE")
            if heartbeat_file_env_path:
                try:
                    # Ensure HEARTBEAT_FILE is an absolute path for reliable dirname
                    abs_heartbeat_path = os.path.abspath(heartbeat_file_env_path)
                    ipc_dir = os.path.dirname(abs_heartbeat_path)
                    self._generated_ipc_file_path = os.path.join(ipc_dir, WANDB_IPC_FILENAME)

                    os.makedirs(ipc_dir, exist_ok=True)  # Ensure directory exists

                    ipc_data = {
                        "run_id": self.run.id,
                        "project": self.run.project,
                        "entity": self.run.entity,
                        "name": self.run.name,
                    }
                    with open(self._generated_ipc_file_path, "w") as f:
                        json.dump(ipc_data, f)
                    logger.info(f"W&B IPC data written to: {self._generated_ipc_file_path}")
                except Exception as e:
                    logger.error(
                        f"Failed to write W&B IPC file alongside heartbeat file ({heartbeat_file_env_path}): {e}",
                        exc_info=True,
                    )
                    self._generated_ipc_file_path = None  # Mark as not generated
            else:
                logger.info("HEARTBEAT_FILE env var not set. Cannot write W&B IPC file for heartbeat monitor.")
            # --- End File-based IPC ---

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
                logger.info(f"Starting W&B cleanup for run: {run.id}")
                wandb.finish()
                logger.info("W&B cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during W&B cleanup: {str(e)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("WandbContext.__exit__ called")
        self.cleanup_run(self.run)
        logger.info("WandbContext.__exit__ completed")
        # No explicit cleanup of the IPC file as per user preference (it's co-located with heartbeat or not written)
