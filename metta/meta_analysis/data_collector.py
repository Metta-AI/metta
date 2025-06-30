"""Data collection utilities for meta-analysis training curve prediction."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.eval.eval_stats_db import EvalStatsDB

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """Collects training data from wandb artifacts for meta-analysis."""

    def __init__(self, wandb_entity: str, wandb_project: str):
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.api = wandb.Api()

    def collect_training_runs(
        self,
        run_filters: Optional[Dict] = None,
        max_runs: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Collect training runs from wandb with their configs and curves.

        Args:
            run_filters: Additional wandb filters
            max_runs: Maximum number of runs to collect
            start_date: Start date filter (ISO format: YYYY-MM-DD)
            end_date: End date filter (ISO format: YYYY-MM-DD)
        """

        # Build filters
        filters = run_filters or {}

        # Add date filters if provided
        if start_date:
            filters["created_at"] = {"$gte": start_date}
        if end_date:
            if "created_at" in filters:
                filters["created_at"]["$lte"] = end_date
            else:
                filters["created_at"] = {"$lte": end_date}

        # Query wandb runs
        if start_date or end_date:
            date_range = f"from {start_date or 'beginning'} to {end_date or 'now'}"
            logger.info(f"Filtering runs {date_range}")

        runs = self.api.runs(f"{self.wandb_entity}/{self.wandb_project}", filters=filters)
        runs = list(runs)[:max_runs]

        training_data = []

        for run in runs:
            try:
                # Extract run data
                run_data = self._extract_run_data(run)
                if run_data:
                    training_data.append(run_data)
                    logger.info(f"Collected data from run: {run.name}")

            except Exception as e:
                logger.warning(f"Failed to collect data from run {run.name}: {e}")
                continue

        logger.info(f"Collected {len(training_data)} training runs")
        return training_data

    def _extract_run_data(self, run) -> Optional[Dict]:
        """Extract training data from a single wandb run."""

        # Get run config
        config = run.config
        if not config:
            return None

        # Extract environment config
        env_config = self._extract_env_config(config)
        if not env_config:
            return None

        # Extract agent config
        agent_config = self._extract_agent_config(config)
        if not agent_config:
            return None

        # Extract training curve (hearts vs timesteps)
        training_curve = self._extract_training_curve(run)
        if not training_curve:
            return None

        return {
            "run_id": run.id,
            "run_name": run.name,
            "env_config": env_config,
            "agent_config": agent_config,
            "training_curve": training_curve,
            "final_performance": run.summary.get("overview/reward", 0.0)
        }

    def _extract_env_config(self, config: Dict) -> Optional[Dict]:
        """Extract environment configuration parameters."""

        env_config = {}

        # Extract from trainer config
        if "trainer" in config:
            trainer = config["trainer"]

            # Environment size parameters
            if "env_overrides" in trainer:
                env_overrides = trainer["env_overrides"]
                if "game" in env_overrides:
                    game = env_overrides["game"]
                    env_config.update({
                        "max_steps": game.get("max_steps", 1000),
                        "num_agents": game.get("num_agents", 1),
                    })

                    # Map builder parameters
                    if "map_builder" in game:
                        map_builder = game["map_builder"]
                        env_config.update({
                            "map_width": map_builder.get("width", 20),
                            "map_height": map_builder.get("height", 20),
                            "num_rooms": map_builder.get("num_rooms", 1),
                        })

                        # Object counts
                        if "objects" in map_builder:
                            objects = map_builder["objects"]
                            env_config.update({
                                "num_altars": objects.get("altar", 0),
                                "num_mines": objects.get("mine_red", 0),
                                "num_generators": objects.get("generator_red", 0),
                                "num_walls": objects.get("wall", 0),
                            })

        # Extract from curriculum config
        if "curriculum" in config.get("trainer", {}):
            curriculum_path = config["trainer"]["curriculum"]
            # Could parse curriculum config files here for more parameters

        return env_config if env_config else None

    def _extract_agent_config(self, config: Dict) -> Optional[Dict]:
        """Extract agent hyperparameters."""

        agent_config = {}

        # Extract from trainer config
        if "trainer" in config:
            trainer = config["trainer"]

            # Learning parameters
            agent_config.update({
                "learning_rate": trainer.get("optimizer", {}).get("learning_rate", 0.001),
                "batch_size": trainer.get("batch_size", 524288),
                "minibatch_size": trainer.get("minibatch_size", 16384),
                "gamma": trainer.get("gamma", 0.977),
                "gae_lambda": trainer.get("gae_lambda", 0.916),
                "clip_coef": trainer.get("clip_coef", 0.1),
                "ent_coef": trainer.get("ent_coef", 0.0021),
                "vf_coef": trainer.get("vf_coef", 0.44),
            })

        # Extract from agent config
        if "agent" in config:
            agent = config["agent"]

            # Network architecture parameters
            if "components" in agent:
                components = agent["components"]

                # Core network size
                if "_core_" in components:
                    core = components["_core_"]
                    agent_config.update({
                        "hidden_size": core.get("output_size", 128),
                        "num_layers": core.get("nn_params", {}).get("num_layers", 2),
                    })

                # CNN parameters
                for comp_name, comp in components.items():
                    if "cnn" in comp_name.lower():
                        agent_config.update({
                            "cnn_channels": comp.get("nn_params", {}).get("out_channels", 64),
                            "cnn_kernel_size": comp.get("nn_params", {}).get("kernel_size", 3),
                        })
                        break

        return agent_config if agent_config else None

    def _extract_training_curve(self, run) -> Optional[np.ndarray]:
        """Extract training curve (hearts vs timesteps) from wandb history."""

        try:
            # Get training history
            history = run.history()
            if history.empty:
                return None

            # Extract reward/hearts data
            reward_cols = [col for col in history.columns if "reward" in col.lower() or "heart" in col.lower()]

            if not reward_cols:
                return None

            # Use the first available reward column
            reward_col = reward_cols[0]

            # Get timesteps and rewards
            timesteps = history.get("metric/agent_step", history.index)
            rewards = history[reward_col]

            # Remove NaN values
            valid_mask = ~(np.isnan(timesteps) | np.isnan(rewards))
            timesteps = timesteps[valid_mask]
            rewards = rewards[valid_mask]

            if len(timesteps) < 10:  # Need minimum data points
                return None

            # Normalize timesteps to [0, 1] and sample at regular intervals
            normalized_timesteps = np.linspace(0, 1, 100)
            normalized_rewards = np.interp(normalized_timesteps,
                                         (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min()),
                                         rewards)

            return normalized_rewards

        except Exception as e:
            logger.warning(f"Failed to extract training curve: {e}")
            return None

    def save_dataset(self, training_data: List[Dict], output_path: str):
        """Save collected dataset to file."""

        # Convert to DataFrame-friendly format
        dataset = []
        for data in training_data:
            row = {
                "run_id": data["run_id"],
                "run_name": data["run_name"],
                "final_performance": data["final_performance"],
                **data["env_config"],
                **data["agent_config"],
                "training_curve": json.dumps(data["training_curve"].tolist())
            }
            dataset.append(row)

        df = pd.DataFrame(dataset)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")

        return df
