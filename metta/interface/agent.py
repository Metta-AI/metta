"""Agent factory for creating Metta agents with a clean API."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

from metta.agent.metta_agent import MettaAgent
from metta.mettagrid import MettaGridEnv

if TYPE_CHECKING:
    from .environment import Environment

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
    """Factory class for creating Metta agents without Hydra dependencies."""

    def __new__(
        cls,
        env: "Environment",
        config: Optional[DictConfig] = None,
        device: str = "cuda",
    ) -> MettaAgent:
        """Create a new MettaAgent instance.

        Args:
            env: Environment to create agent for (must have a MettaGridEnv driver)
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


def create_or_load_agent(
    env: "Environment",
    run_dir: str,
    policy_store: Any,  # PolicyStore
    trainer_config: Any,  # TrainerConfig
    device: str | torch.device = "cuda",
    is_master: bool = True,
    rank: int = 0,
) -> tuple[MettaAgent, Any, int, int, Any]:  # Returns (agent, policy_record, agent_step, epoch, checkpoint)
    """Create a new agent or load from checkpoint/initial policy.

    This helper function encapsulates the full logic for agent creation/loading,
    including checkpoint handling, following the same pattern as trainer.py.

    Args:
        env: Environment (must have MettaGridEnv driver)
        run_dir: Run directory containing checkpoints
        policy_store: PolicyStore for loading saved policies
        trainer_config: TrainerConfig with checkpoint/initial_policy settings
        device: Device to use
        is_master: Whether this is the master process
        rank: Process rank for distributed training

    Returns:
        Tuple of (agent, policy_record, agent_step, epoch, checkpoint)
    """
    from metta.rl.policy_management import maybe_load_checkpoint

    # Get the MettaGridEnv
    metta_grid_env = env.driver_env
    assert isinstance(metta_grid_env, MettaGridEnv)

    # Load checkpoint and policy
    checkpoint, policy_record, agent_step, epoch = maybe_load_checkpoint(
        run_dir=run_dir,
        policy_store=policy_store,
        trainer_cfg=trainer_config,
        metta_grid_env=metta_grid_env,
        cfg=DictConfig(
            {
                "device": str(device),
                "run": os.path.basename(run_dir),
                "run_dir": run_dir,
                "agent": _get_default_agent_config(str(device))["agent"],
            }
        ),
        device=device,
        is_master=is_master,
        rank=rank,
    )

    if policy_record is not None:
        # Use loaded policy
        agent = policy_record.policy
        logger.info(f"Loaded agent from {policy_record.uri}")

        # Initialize to environment (handles feature remapping)
        features = metta_grid_env.get_observation_features()
        agent.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)
    else:
        # Create new agent
        agent = Agent(env, device=str(device))
        logger.info("Created new agent")

    return agent, policy_record, agent_step, epoch, checkpoint
