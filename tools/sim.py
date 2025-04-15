import logging
from dataclasses import dataclass
from typing import List

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.sim.eval_stats_logger import EvalStatsLogger
from metta.sim.simulation import SimulationSuite
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext
from dataclasses import dataclass, fields, is_dataclass
from typing import Type, TypeVar


@dataclass
class SimJob:
    policy_uris: List[str] = MISSING
    eval_db_uri: str = MISSING
    simulation_suite: SimulationSuiteConfig = MISSING
    selector_type: str = "latest"


T = TypeVar("T")


def convert_to_dataclass(cls: Type[T], config_dict: dict, strict: bool = False, allow_missing: bool = False) -> T:
    """
    Convert a dictionary to a dataclass instance with validation.

    Args:
        cls: The dataclass type to convert to
        config_dict: The dictionary containing configuration values
        strict: If True, errors on extra keys not in the dataclass
        allow_missing: If True, allows MISSING values to remain

    Returns:
        An instance of the dataclass type
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")

    # Get all field names from the dataclass
    field_names = {f.name for f in fields(cls)}

    # Check for extra keys
    extra_keys = set(config_dict.keys()) - field_names
    if extra_keys:
        if strict:
            raise ValueError(f"Extra keys found: {extra_keys}")
        else:
            print(f"Warning: Ignoring extra keys: {extra_keys}")

    # Filter out extra keys
    filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

    # Create instance
    instance = cls(**filtered_dict)

    # Validate no MISSING values unless allowed
    if not allow_missing:
        for field in fields(cls):
            value = getattr(instance, field.name)
            if value is MISSING:
                raise ValueError(f"Field '{field.name}' is required but missing")

    return instance


def simulate_policy(sim_job: SimJob, policy_uri: str, cfg: DictConfig, wandb_run):
    logger = logging.getLogger("metta.tools.sim")
    # TODO: Remove dependence on cfg in PolicyStore
    policy_store = PolicyStore(cfg, wandb_run)
    policy_prs = policy_store.policies(policy_uri, sim_job.selector_type)
    # For each checkpoint of the policy, simulate
    for pr in policy_prs:
        logger.info(f"Evaluating policy {pr.uri}")
        sim = SimulationSuite(sim_job.simulation_suite, policy_store, pr)
        stats = sim.simulate()
        stats_logger = EvalStatsLogger(sim_job.simulation_suite, wandb_run)
        stats_logger.log(stats)
        logger.info(f"Evaluation complete for policy {pr.uri}; logging stats")


@hydra.main(version_base=None, config_path="../configs", config_name="sim")
def main(cfg: DictConfig):
    setup_mettagrid_environment(cfg)
    sim_job = convert_to_dataclass(SimJob, cfg.sim_job)
    with WandbContext(cfg) as wandb_run:
        for policy_uri in sim_job.policy_uris:
            simulate_policy(sim_job, policy_uri, cfg, wandb_run)


if __name__ == "__main__":
    main()
