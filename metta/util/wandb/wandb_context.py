import copy
import logging
import os
import socket

import pkg_resources
import requests
import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


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

        check_wandb_version()

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
