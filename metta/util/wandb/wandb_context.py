import copy
import logging
import os
import socket

import pkg_resources
import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def check_wandb_version():
    try:
        # Get the installed wandb version
        wandb_version = pkg_resources.get_distribution("wandb").version

        minor_version = int(wandb_version.split(".")[1])

        # Check if the major version is less than 19
        if minor_version < 19:
            print(f"ERROR: Your wandb version ({wandb_version}) is outdated.")
            print("Please update to wandb version 0.19 or later using:")
            print("    pip install --upgrade wandb")
            return False

        return True

    except pkg_resources.DistributionNotFound:
        print("ERROR: wandb package is not installed.")
        print("Please install wandb using:")
        print("    pip install wandb")
        return False


class WandbContext:
    def __init__(self, cfg, job_type=None, resume=True, name=None, run_id=None, data_dir=None, timeout=30):
        self.cfg = cfg
        self.resume = resume
        self.name = name or cfg.wandb.name
        self.run_id = cfg.wandb.run_id or self.cfg.run or wandb.util.generate_id()
        self.run = None
        self.data_dir = data_dir or self.cfg.run_dir
        self.job_type = job_type
        self.timeout = timeout  # Add configurable timeout (wandb default is 90 seconds)
        self.wandb_host = "api.wandb.ai"
        self.wandb_port = 443

        if not check_wandb_version():
            logger.warning("Please upgrade wandb to >= 19")

    def __enter__(self) -> wandb.apis.public.Run:
        if not self.cfg.wandb.enabled:
            assert not self.cfg.wandb.track, "wandb.track won't work if wandb.enabled is False"
            return None

        # Check internet connection before proceeding
        try:
            socket.setdefaulttimeout(5)  # Set a 5-second timeout for the connection check
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.wandb_host, self.wandb_port))
            logger.info(f"Connection to {self.wandb_host} verified")
        except Exception as ex:
            logger.warning(f"No connection to {self.wandb_host} servers detected: {str(ex)}")
            logger.info("Continuing without W&B logging")
            return None

        cfg = copy.deepcopy(self.cfg)

        try:
            logger.info(f"Initializing W&B run with timeout={self.timeout}s")

            try:
                self.run = wandb.init(
                    id=self.run_id,
                    job_type=self.job_type,
                    project=self.cfg.wandb.project,
                    entity=self.cfg.wandb.entity,
                    config=OmegaConf.to_container(cfg, resolve=False),
                    group=self.cfg.wandb.group,
                    allow_val_change=True,
                    name=self.name,
                    monitor_gym=True,
                    save_code=True,
                    resume=self.resume,
                    tags=["user:" + os.environ.get("METTA_USER", "unknown")],
                    settings=wandb.Settings(quiet=True, init_timeout=self.timeout),
                )

                # Save config and set up file syncing only if wandb init succeeded
                OmegaConf.save(cfg, os.path.join(self.data_dir, "config.yaml"))
                wandb.save(os.path.join(self.data_dir, "*.log"), base_path=self.data_dir, policy="live")
                wandb.save(os.path.join(self.data_dir, "*.yaml"), base_path=self.data_dir, policy="live")
                logger.info(f"Successfully initialized W&B run: {self.run.name} ({self.run.id})")

            except TimeoutError:
                logger.warning(f"W&B initialization timed out after {self.timeout}s")
                logger.info("Continuing without W&B logging")
                self.run = None

        except wandb.errors.CommError as e:
            logger.error(f"W&B initialization failed: {str(e)}")
            logger.info("Continuing without W&B logging")
            self.run = None

        except Exception as e:
            logger.error(f"Unexpected error during W&B initialization: {str(e)}")
            logger.info("Continuing without W&B logging")
            self.run = None

        return self.run

    @staticmethod
    def make_run(cfg, resume=True, name=None):
        return WandbContext(cfg, resume=resume, name=name).__enter__()

    @staticmethod
    def cleanup_run(run):
        if run:
            try:
                wandb.finish()
            except Exception as e:
                logger.error(f"Error during W&B cleanup: {str(e)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_run(self.run)
