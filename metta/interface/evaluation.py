"""Evaluation and replay functionality for Metta."""

import logging
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig

from .environment import _get_default_env_config

logger = logging.getLogger(__name__)


def create_evaluation_config_suite() -> SimulationSuiteConfig:
    """Create evaluation suite configuration with pre-built configs to bypass Hydra.

    This creates the standard navigation evaluation suite used by the training system,
    but with pre-built configs that don't require Hydra to be initialized.

    Returns:
        SimulationSuiteConfig with navigation tasks
    """
    # Create pre-built navigation evaluation configs
    base_nav_config = _get_default_env_config()
    base_nav_config["sampling"] = 0  # Disable sampling for evaluation
    base_nav_config["game"]["track_exploration"] = True  # Enable exploration tracking for evaluation

    # Create evaluation configs for different terrain sizes
    simulations = {}

    # Small terrain evaluation
    simulations["navigation/terrain_small"] = {
        "env": "/env/mettagrid/navigation/training/terrain_from_numpy",
        "num_episodes": 5,
        "max_time_s": 30,
        "env_overrides": {
            "game": {"map_builder": {"room": {"dir": "varied_terrain/balanced_small"}}},
            "_pre_built_env_config": DictConfig(base_nav_config.copy()),
        },
    }

    # Medium terrain evaluation
    simulations["navigation/terrain_medium"] = {
        "env": "/env/mettagrid/navigation/training/terrain_from_numpy",
        "num_episodes": 5,
        "max_time_s": 30,
        "env_overrides": {
            "game": {"map_builder": {"room": {"dir": "varied_terrain/balanced_medium"}}},
            "_pre_built_env_config": DictConfig(base_nav_config.copy()),
        },
    }

    # Large terrain evaluation
    simulations["navigation/terrain_large"] = {
        "env": "/env/mettagrid/navigation/training/terrain_from_numpy",
        "num_episodes": 5,
        "max_time_s": 30,
        "env_overrides": {
            "game": {"map_builder": {"room": {"dir": "varied_terrain/balanced_large"}}},
            "_pre_built_env_config": DictConfig(base_nav_config.copy()),
        },
    }

    # Create suite config
    evaluation_config = SimulationSuiteConfig(
        name="evaluation",
        simulations=simulations,
        num_episodes=10,  # Will be overridden by individual configs
        env_overrides={},  # Suite-level overrides
    )

    return evaluation_config


def create_replay_config(terrain_dir: str = "varied_terrain/balanced_medium") -> SingleEnvSimulationConfig:
    """Create a pre-built replay configuration to bypass Hydra.

    Args:
        terrain_dir: Directory for terrain maps (default: varied_terrain/balanced_medium)

    Returns:
        SingleEnvSimulationConfig with pre-built config attached
    """
    # Create pre-built navigation config for replay
    replay_config = _get_default_env_config()
    replay_config["sampling"] = 0  # Disable sampling for replay

    # Create simulation config with pre-built config in env_overrides
    replay_sim_config = SingleEnvSimulationConfig(
        env="/env/mettagrid/navigation/training/terrain_from_numpy",
        num_episodes=1,
        max_time_s=60,
        env_overrides={
            "game": {"map_builder": {"room": {"dir": terrain_dir}}},
            "_pre_built_env_config": DictConfig(replay_config),
        },
    )

    return replay_sim_config


def evaluate_policy_suite(
    policy_record: Any,
    policy_store: Any,
    evaluation_config: SimulationSuiteConfig,
    device: torch.device,
    vectorization: str = "serial",
    stats_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """Run evaluation suite on a policy and return scores.

    This is a simplified version of the evaluation logic that both run.py
    and trainer.py can use without needing stats_client or wandb.

    Args:
        policy_record: The policy record to evaluate
        policy_store: PolicyStore instance
        evaluation_config: SimulationSuiteConfig with evaluation tasks
        device: Device to run on
        vectorization: Vectorization mode
        stats_dir: Directory for stats (optional)
        logger: Logger instance (optional)

    Returns:
        Dictionary of evaluation scores
    """
    from metta.common.util.heartbeat import record_heartbeat
    from metta.eval.eval_stats_db import EvalStatsDB
    from metta.sim.simulation_suite import SimulationSuite

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Evaluating policy: {policy_record.uri}")

    # Run evaluation suite
    sim_suite = SimulationSuite(
        config=evaluation_config,
        policy_pr=policy_record,
        policy_store=policy_store,
        device=device,
        vectorization=vectorization,
        stats_dir=stats_dir or "/tmp/stats",
        stats_client=None,
        stats_epoch_id=None,
        wandb_policy_name=None,
    )

    results = sim_suite.simulate()
    stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
    logger.info("Evaluation complete")

    # Build evaluation metrics
    eval_scores = {}
    categories = set()
    for sim_name in evaluation_config.simulations.keys():
        categories.add(sim_name.split("/")[0])

    for category in categories:
        score = stats_db.get_average_metric_by_filter("reward", policy_record, f"sim_name LIKE '%{category}%'")
        logger.info(f"{category} score: {score}")
        record_heartbeat()
        if score is not None:
            eval_scores[f"{category}/score"] = score

    # Get detailed per-simulation scores
    all_scores = stats_db.simulation_scores(policy_record, "reward")
    for (_, sim_name, _), score in all_scores.items():
        category = sim_name.split("/")[0]
        sim_short_name = sim_name.split("/")[-1]
        eval_scores[f"{category}/{sim_short_name}"] = score

    stats_db.close()
    return eval_scores


def generate_replay_simple(
    policy_record: Any,
    policy_store: Any,
    replay_config: SingleEnvSimulationConfig,
    device: torch.device,
    vectorization: str = "serial",
    replay_dir: str = "./replays",
    epoch: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Generate a replay and return the player URL.

    This is a simplified version of replay generation that both run.py
    and trainer.py can use.

    Args:
        policy_record: The policy record to replay
        policy_store: PolicyStore instance
        replay_config: SingleEnvSimulationConfig for replay
        device: Device to run on
        vectorization: Vectorization mode
        replay_dir: Directory to save replays
        epoch: Optional epoch number for naming
        logger: Logger instance (optional)

    Returns:
        Player URL if replay was generated, None otherwise
    """
    from metta.sim.simulation import Simulation

    if logger is None:
        logger = logging.getLogger(__name__)

    replay_name = f"replay_{epoch}" if epoch is not None else "replay"
    logger.info(f"Generating {replay_name}")

    replay_simulator = Simulation(
        name=replay_name,
        config=replay_config,
        policy_pr=policy_record,
        policy_store=policy_store,
        device=device,
        vectorization=vectorization,
        replay_dir=replay_dir,
    )

    results = replay_simulator.simulate()

    # Get replay URLs from the database
    replay_urls = results.stats_db.get_replay_urls()
    player_url = None
    if replay_urls:
        replay_url = replay_urls[0]
        player_url = f"https://metta-ai.github.io/metta/?replayUrl={replay_url}"
        logger.info(f"Replay available at: {player_url}")

    results.stats_db.close()
    return player_url
