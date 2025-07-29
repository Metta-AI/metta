"""Weights & Biases integration utilities."""

import logging
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbContext
from metta.rl.wandb import upload_env_configs

logger = logging.getLogger(__name__)


def setup_wandb_context(
    wandb_config: Optional[DictConfig],
    global_config: Optional[DictConfig],
    is_master: bool,
) -> Optional[WandbContext]:
    """Set up WandB context for master rank.

    Args:
        wandb_config: WandB configuration
        global_config: Global configuration
        is_master: Whether this is the master rank

    Returns:
        WandbContext or None if not master or config missing
    """
    if not is_master or not wandb_config or not global_config:
        return None

    return WandbContext(wandb_config, global_config)


def upload_environment_configs(
    curriculum: Any,
    wandb_run: Any,
    is_master: bool,
) -> None:
    """Upload environment configurations to WandB.

    Args:
        curriculum: Curriculum object with environment configs
        wandb_run: WandB run object
        is_master: Whether this is the master rank
    """
    if not is_master or not wandb_run or not curriculum:
        return

    if hasattr(curriculum, "get_env_cfg_by_bucket"):
        env_configs = curriculum.get_env_cfg_by_bucket()
        upload_env_configs(env_configs=env_configs, wandb_run=wandb_run)
        logger.info("Uploaded environment configs to WandB")


def upload_policy_artifact(
    wandb_run: Any,
    policy_store: PolicyStore,
    policy_record: Any,
    force: bool = False,
) -> Optional[str]:
    """Upload policy to WandB as artifact.

    Args:
        wandb_run: WandB run object
        policy_store: Policy store
        policy_record: Policy record to upload
        force: Force upload even if already uploaded

    Returns:
        WandB policy name or None if failed
    """
    if not wandb_run or not policy_record:
        return None

    try:
        wandb_policy_name = policy_store.add_to_wandb_run(wandb_run.id, policy_record, force=force)
        logger.info(f"Uploaded policy to wandb: {wandb_policy_name}")
        return wandb_policy_name
    except Exception as e:
        logger.warning(f"Failed to upload policy to wandb: {e}")
        return None


def create_wandb_config(
    trainer_config: Any,
    agent_config: Any,
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create configuration dictionary for WandB.

    Args:
        trainer_config: Trainer configuration
        agent_config: Agent configuration
        run_config: Additional run configuration

    Returns:
        Configuration dictionary for WandB
    """
    config = {}

    # Add trainer config
    if hasattr(trainer_config, "model_dump"):
        config["trainer"] = trainer_config.model_dump()
    elif isinstance(trainer_config, DictConfig):
        config["trainer"] = OmegaConf.to_container(trainer_config, resolve=True)
    else:
        config["trainer"] = trainer_config

    # Add agent config
    if agent_config:
        if hasattr(agent_config, "model_dump"):
            config["agent"] = agent_config.model_dump()
        elif isinstance(agent_config, DictConfig):
            config["agent"] = OmegaConf.to_container(agent_config, resolve=True)
        else:
            config["agent"] = agent_config

    # Add run config
    config.update(run_config)

    return config


def should_upload_to_wandb(
    epoch: int,
    wandb_checkpoint_interval: int,
    is_master: bool,
    wandb_run: Any,
) -> bool:
    """Check if policy should be uploaded to WandB.

    Args:
        epoch: Current epoch
        wandb_checkpoint_interval: Interval for WandB uploads
        is_master: Whether this is master rank
        wandb_run: WandB run object

    Returns:
        True if should upload
    """
    if not is_master or not wandb_run or wandb_checkpoint_interval <= 0:
        return False
    return epoch % wandb_checkpoint_interval == 0
