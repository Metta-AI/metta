from typing import Any, ClassVar

from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.common.wandb.wandb_context import WandbConfig
from metta.rl.trainer_config import TrainerConfig, parse_trainer_config
from metta.sim.simulation_config import SimulationSuiteConfig


class TrainJobInnerConfig(BaseModelWithForbidExtra):
    evals: SimulationSuiteConfig
    map_preview_uri: str | None = None


class TrainJobConfig(BaseModelWithForbidExtra):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # Allow DictConfig
    )
    # Core configuration
    run: str
    run_dir: str
    data_dir: str
    policy_uri: str
    device: str = "cuda"
    seed: int = 0

    # Component configurations
    trainer: TrainerConfig
    wandb: WandbConfig
    train_job: TrainJobInnerConfig

    # Additional settings
    policy_cache_size: int = 10
    stats_server_uri: str = "https://api.observatory.softmax-research.net"
    torch_deterministic: bool = True
    vectorization: str = "multiprocessing"
    dist_cfg_path: str | None = None
    cmd: str = "train"

    # Keep these as DictConfig for now until they're typed
    agent: DictConfig | None = None
    pytorch: DictConfig | None = None
    sim: Any | None = None  # This is referenced but not used in the typed config


def parse_train_job_config(cfg: DictConfig | dict) -> TrainJobConfig:
    """Parse and validate a train job configuration.

    Args:
        cfg: Raw configuration from Hydra or dict

    Returns:
        Validated TrainJobConfig instance
    """
    # Convert to dict if needed
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("Configuration must be a dict")
    else:
        cfg_dict = cfg

    # Parse nested trainer config if present
    if "trainer" in cfg_dict:
        # The trainer config parser handles its own runtime paths
        trainer_cfg = parse_trainer_config(cfg)
        cfg_dict["trainer"] = trainer_cfg

    # Parse wandb config
    if "wandb" in cfg_dict:
        wandb_dict = cfg_dict["wandb"]
        if isinstance(wandb_dict, dict):
            # WandbConfig is a Union type, so we need to validate the dict directly
            from metta.common.wandb.wandb_context import WandbConfigOff, WandbConfigOn

            if wandb_dict.get("enabled", True):
                wandb_cfg = WandbConfigOn.model_validate(wandb_dict)
            else:
                wandb_cfg = WandbConfigOff.model_validate(wandb_dict)
            cfg_dict["wandb"] = wandb_cfg
        else:
            # Handle other types (e.g., already validated configs)
            cfg_dict["wandb"] = wandb_dict

    # Parse train_job inner config
    if "train_job" in cfg_dict:
        train_job_dict = cfg_dict["train_job"]
        if isinstance(train_job_dict, dict):
            # Handle evals which should be SimulationSuiteConfig
            if "evals" in train_job_dict and not isinstance(train_job_dict["evals"], SimulationSuiteConfig):
                # For now, create SimulationSuiteConfig from the dict/DictConfig
                evals_data = train_job_dict["evals"]
                if isinstance(evals_data, DictConfig):
                    evals_dict = OmegaConf.to_container(evals_data, resolve=True)
                else:
                    evals_dict = evals_data
                train_job_dict["evals"] = SimulationSuiteConfig.model_validate(evals_dict)
            cfg_dict["train_job"] = TrainJobInnerConfig.model_validate(train_job_dict)

    # Keep agent and pytorch as DictConfig for backward compatibility
    if "agent" in cfg_dict and not isinstance(cfg_dict["agent"], DictConfig):
        cfg_dict["agent"] = DictConfig(cfg_dict["agent"])
    if "pytorch" in cfg_dict and cfg_dict["pytorch"] is not None:
        if not isinstance(cfg_dict["pytorch"], DictConfig):
            cfg_dict["pytorch"] = DictConfig(cfg_dict["pytorch"])

    return TrainJobConfig.model_validate(cfg_dict)
