"""Agent factory for Metta."""

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig

from metta.agent.metta_agent import MettaAgent
from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger(__name__)


def _get_default_agent_config(device: str = "cuda") -> DictConfig:
    """Get default agent configuration based on fast.yaml architecture."""
    return DictConfig(
        {
            "device": device,
            "agent": {
                "clip_range": 0,
                "analyze_weights_interval": 300,
                "l2_init_weight_update_interval": 0,
                "observations": {"obs_key": "grid_obs"},
                "components": {
                    "_obs_": {
                        "_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper",
                        "sources": None,
                    },
                    "obs_normalizer": {
                        "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                        "sources": [{"name": "_obs_"}],
                    },
                    "cnn1": {
                        "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                        "sources": [{"name": "obs_normalizer"}],
                        "nn_params": {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 1,
                        },
                    },
                    "cnn2": {
                        "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                        "sources": [{"name": "cnn1"}],
                        "nn_params": {
                            "out_channels": 64,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 1,
                        },
                    },
                    "obs_flattener": {
                        "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                        "sources": [{"name": "cnn2"}],
                    },
                    "encoded_obs": {
                        "_target_": "metta.agent.lib.nn_layer_library.Linear",
                        "sources": [{"name": "obs_flattener"}],
                        "nn_params": {"out_features": 512},
                    },
                    "_core_": {
                        "_target_": "metta.agent.lib.lstm.LSTM",
                        "sources": [{"name": "encoded_obs"}],
                        "output_size": 512,
                        "nn_params": {
                            "num_layers": 1,
                        },
                    },
                    "_value_": {
                        "_target_": "metta.agent.lib.nn_layer_library.Linear",
                        "sources": [{"name": "_core_"}],
                        "nn_params": {"out_features": 1},
                        "nonlinearity": None,
                    },
                    "actor_1": {
                        "_target_": "metta.agent.lib.nn_layer_library.Linear",
                        "sources": [{"name": "_core_"}],
                        "nn_params": {"out_features": 512},
                    },
                    "_action_embeds_": {
                        "_target_": "metta.agent.lib.action.ActionEmbedding",
                        "sources": None,
                        "nn_params": {
                            "num_embeddings": 100,
                            "embedding_dim": 16,
                        },
                    },
                    "_action_": {
                        "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
                        "sources": [
                            {"name": "actor_1"},
                            {"name": "_action_embeds_"},
                        ],
                    },
                },
            },
        }
    )


class Agent:
    """Factory for creating Metta agents with a clean API.

    This handles agent creation and initialization without Hydra.
    """

    def __new__(
        cls,
        env: Any,  # vecenv wrapper
        config: Optional[DictConfig] = None,
        device: str = "cuda",
    ) -> Any:  # Returns MettaAgent or DistributedMettaAgent
        """Create a Metta agent.

        Args:
            env: Vectorized environment (from Environment factory)
            config: Optional DictConfig with agent configuration. If not provided, uses a default configuration.
            device: Device to use

        Returns:
            MettaAgent instance (or DistributedMettaAgent wrapper if distributed is initialized)
        """
        logger.info("Creating agent...")

        # Use default config if none provided
        if config is None:
            config = _get_default_agent_config(device)

        # Get the actual MettaGridEnv from vecenv wrapper
        metta_grid_env = env.driver_env
        assert isinstance(metta_grid_env, MettaGridEnv)

        # Create observation space matching what make_policy does
        obs_space = gym.spaces.Dict(
            {
                "grid_obs": metta_grid_env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

        # Extract agent config
        agent_cfg = config.agent

        # Create MettaAgent directly without Hydra
        agent = MettaAgent(
            obs_space=obs_space,
            obs_width=metta_grid_env.obs_width,
            obs_height=metta_grid_env.obs_height,
            action_space=metta_grid_env.single_action_space,
            feature_normalizations=metta_grid_env.feature_normalizations,
            global_features=metta_grid_env.global_features,
            device=str(device),
            **agent_cfg,
        )

        # Initialize to environment
        features = metta_grid_env.get_observation_features()
        agent.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)

        return agent
