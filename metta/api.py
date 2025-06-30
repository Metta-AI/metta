"""
Clean API for Metta - provides direct instantiation without Hydra.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# Object type IDs from mettagrid/src/metta/mettagrid/objects/constants.hpp
# These define the type of object in the environment
TYPE_AGENT = 0
TYPE_WALL = 1
TYPE_MINE_RED = 2
TYPE_MINE_BLUE = 3
TYPE_MINE_GREEN = 4
TYPE_GENERATOR_RED = 5
TYPE_GENERATOR_BLUE = 6
TYPE_GENERATOR_GREEN = 7
TYPE_ALTAR = 8
TYPE_ARMORY = 9
TYPE_LASERY = 10
TYPE_LAB = 11
TYPE_FACTORY = 12
TYPE_TEMPLE = 13
TYPE_GENERIC_CONVERTER = 14

# Default training parameters
DEFAULT_PARAMS = {
    "learning_rate": 0.0004573146765703167,
    "batch_size": 262_144,
    "minibatch_size": 16_384,
    "bptt_horizon": 64,
    "update_epochs": 1,
    "max_grad_norm": 0.5,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-12,
    "weight_decay": 0,
    "vf_coef": 0.44,
    "ent_coef": 0.0021,
    "clip_coef": 0.1,
    "vf_clip_coef": 0.1,
    "gamma": 0.977,
    "gae_lambda": 0.916,
    "vtrace_rho_clip": 1.0,
    "vtrace_c_clip": 1.0,
}

# Default environment parameters
DEFAULT_ENV_PARAMS = {
    "width": 15,
    "height": 10,
    "obs_width": 11,
    "obs_height": 11,
    "max_steps": 1000,
    "num_agents": 2,
}


# Data structures for enhanced training state
@dataclass
class TrainingState:
    """Complete training state for checkpointing."""

    epoch: int
    agent_step: int
    total_agent_step: int
    optimizer_state_dict: Dict[str, Any]
    lr_scheduler_state_dict: Optional[Dict[str, Any]]
    policy_path: Optional[str]
    stopwatch_state: Optional[Dict[str, Any]]
    extra_args: Dict[str, Any]

    def save(self, checkpoint_dir: str) -> str:
        """Save training state to checkpoint file."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"training_state_epoch_{self.epoch}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str) -> "TrainingState":
        """Load training state from checkpoint file."""
        with open(path, "rb") as f:
            return pickle.load(f)


# Helper functions for enhanced features
def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_timesteps: int,
    batch_size: int,
    warmup_steps: Optional[int] = None,
    schedule_type: str = "linear",
    anneal_lr: bool = True,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a learning rate scheduler."""
    if not anneal_lr:
        return None

    total_updates = total_timesteps // batch_size

    if schedule_type == "linear":

        def lr_lambda(epoch):
            if warmup_steps and epoch < warmup_steps:
                return epoch / warmup_steps
            # Avoid division by zero
            if total_updates <= (warmup_steps or 0):
                return 1.0
            progress = (epoch - (warmup_steps or 0)) / (total_updates - (warmup_steps or 0))
            return max(0.0, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda)
    elif schedule_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_updates)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for monitoring."""
    grad_norms = []
    param_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
            param_norms.append(param.norm().item())

    if not grad_norms:
        return {}

    return {
        "grad_norm_mean": np.mean(grad_norms),
        "grad_norm_max": np.max(grad_norms),
        "grad_norm_min": np.min(grad_norms),
        "param_norm_mean": np.mean(param_norms),
    }


# Direct instantiation functions


def make_agent(
    obs_space,
    action_space,
    obs_width: int,
    obs_height: int,
    feature_normalizations: Dict[int, float],
    global_features: list,
    device: torch.device,
    obs_key: str = "grid_obs",
    clip_range: float = 0,
    analyze_weights_interval: int = 300,
    l2_init_weight_update_interval: int = 0,
):
    """Create a Metta agent instance directly."""
    from metta.agent.metta_agent import MettaAgent

    # Create agent config directly
    config = {
        "_target_": "metta.agent.metta_agent.MettaAgent",
        "observations": {"obs_key": obs_key},
        "clip_range": clip_range,
        "analyze_weights_interval": analyze_weights_interval,
        "l2_init_weight_update_interval": l2_init_weight_update_interval,
        "components": _get_default_agent_components(),
    }

    return MettaAgent(
        obs_space=obs_space,
        obs_width=obs_width,
        obs_height=obs_height,
        action_space=action_space,
        feature_normalizations=feature_normalizations,
        global_features=global_features,
        device=device,
        **config,
    )


def _get_default_agent_components():
    """Get default agent component configuration."""
    return {
        "_obs_": {"_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper", "sources": None},
        "obs_normalizer": {
            "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
            "sources": [{"name": "_obs_"}],
        },
        "cnn1": {
            "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
            "sources": [{"name": "obs_normalizer"}],
            "nn_params": {"out_channels": 64, "kernel_size": 5, "stride": 3},
        },
        "cnn2": {
            "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
            "sources": [{"name": "cnn1"}],
            "nn_params": {"out_channels": 64, "kernel_size": 3, "stride": 1},
        },
        "obs_flattener": {"_target_": "metta.agent.lib.nn_layer_library.Flatten", "sources": [{"name": "cnn2"}]},
        "fc1": {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "obs_flattener"}],
            "nn_params": {"out_features": 128},
        },
        "encoded_obs": {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "fc1"}],
            "nn_params": {"out_features": 128},
        },
        "_core_": {
            "_target_": "metta.agent.lib.lstm.LSTM",
            "sources": [{"name": "encoded_obs"}],
            "output_size": 128,
            "nn_params": {"num_layers": 2},
        },
        "core_relu": {"_target_": "metta.agent.lib.nn_layer_library.ReLU", "sources": [{"name": "_core_"}]},
        "critic_1": {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "core_relu"}],
            "nn_params": {"out_features": 1024},
            "nonlinearity": "nn.Tanh",
            "effective_rank": True,
        },
        "_value_": {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "critic_1"}],
            "nn_params": {"out_features": 1},
            "nonlinearity": None,
        },
        "actor_1": {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "core_relu"}],
            "nn_params": {"out_features": 512},
        },
        "_action_embeds_": {
            "_target_": "metta.agent.lib.action.ActionEmbedding",
            "sources": None,
            "nn_params": {"num_embeddings": 100, "embedding_dim": 16},
        },
        "_action_": {
            "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
            "sources": [{"name": "actor_1"}, {"name": "_action_embeds_"}],
        },
    }


def make_optimizer(parameters, **kwargs) -> torch.optim.Optimizer:
    """Create an optimizer directly."""
    params = {**DEFAULT_PARAMS, **kwargs}
    opt_type = params.pop("type", "adam")

    if opt_type == "adam":
        return torch.optim.Adam(
            parameters,
            lr=params["learning_rate"],
            betas=(params["beta1"], params["beta2"]),
            eps=params["eps"],
            weight_decay=params["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def make_experience_buffer(
    total_agents: int,
    batch_size: int,
    obs_space,
    atn_space,
    device: torch.device,
    hidden_size: int,
    **kwargs,
):
    """Create an experience buffer directly."""
    from metta.rl.experience import Experience

    params = {
        "bptt_horizon": DEFAULT_PARAMS["bptt_horizon"],
        "minibatch_size": DEFAULT_PARAMS["minibatch_size"],
        "max_minibatch_size": DEFAULT_PARAMS["minibatch_size"],
        "cpu_offload": False,
        "num_lstm_layers": 2,
        "agents_per_batch": None,
        **kwargs,
    }

    return Experience(
        total_agents=total_agents,
        batch_size=batch_size,
        obs_space=obs_space,
        atn_space=atn_space,
        device=device,
        hidden_size=hidden_size,
        **params,
    )


def make_loss_module(policy: torch.nn.Module, **kwargs):
    """Create a PPO loss module directly."""
    from metta.rl.objectives import ClipPPOLoss

    params = {
        "vf_coef": DEFAULT_PARAMS["vf_coef"],
        "ent_coef": DEFAULT_PARAMS["ent_coef"],
        "clip_coef": DEFAULT_PARAMS["clip_coef"],
        "vf_clip_coef": DEFAULT_PARAMS["vf_clip_coef"],
        "gamma": DEFAULT_PARAMS["gamma"],
        "gae_lambda": DEFAULT_PARAMS["gae_lambda"],
        "vtrace_rho_clip": DEFAULT_PARAMS["vtrace_rho_clip"],
        "vtrace_c_clip": DEFAULT_PARAMS["vtrace_c_clip"],
        "norm_adv": True,
        "clip_vloss": True,
        "l2_reg_loss_coef": 0.0,
        "l2_init_loss_coef": 0.0,
        "kickstarter": None,
        **kwargs,
    }

    return ClipPPOLoss(policy=policy, **params)


def make_vecenv(
    env_config: Dict[str, Any],
    num_envs: int = 16,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    device: str = "cpu",
    zero_copy: bool = True,
    vectorization: str = "serial",
):
    """Create a vectorized environment directly."""
    from metta.mettagrid.curriculum.core import SingleTaskCurriculum
    from metta.rl.vecenv import make_vecenv

    curriculum = SingleTaskCurriculum("task", DictConfig(env_config))

    return make_vecenv(
        curriculum=curriculum,
        vectorization=vectorization,
        num_envs=num_envs,
        batch_size=num_envs // num_workers if num_workers > 1 else None,
        num_workers=num_workers,
        zero_copy=zero_copy,
        is_training=True,
    )


def env(num_agents: int = None, **kwargs) -> Dict[str, Any]:
    """Create a default MetaGrid environment configuration."""
    params = {**DEFAULT_ENV_PARAMS, **kwargs}
    if num_agents is not None:
        params["num_agents"] = num_agents

    return {
        "sampling": 0,
        "desync_episodes": False,
        "replay_level_prob": 0.0,
        "game": {
            "num_agents": params["num_agents"],
            "obs_width": params["obs_width"],
            "obs_height": params["obs_height"],
            "num_observation_tokens": 200,
            "max_steps": params["max_steps"],
            "inventory_item_names": ["ore.red", "battery.red", "heart", "laser", "armor"],
            "diversity_bonus": {"enabled": False, "similarity_coef": 0.5, "diversity_coef": 0.5},
            "agent": {
                "default_item_max": 50,
                "heart_max": 255,
                "freeze_duration": 10,
                "rewards": {
                    "action_failure_penalty": 0,
                    "ore.red": 0.01,
                    "battery.red": 0.02,
                    "heart": 1,
                    "ore.red_max": 10,
                    "battery.red_max": 10,
                    "heart_max": 1000,
                },
            },
            "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
            "objects": _get_default_objects(),
            "actions": _get_default_actions(),
            "reward_sharing": {"groups": {}},
            "map_builder": {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": params["width"],
                "height": params["height"],
                "border_width": 2,
                "agents": params["num_agents"],
                "objects": {"mine_red": 2, "generator_red": 1, "altar": 1, "wall": 5, "block": 3},
            },
        },
    }


def _get_default_objects():
    """Get default game objects configuration."""
    return {
        "altar": {
            "type_id": TYPE_ALTAR,
            "input_battery.red": 1,
            "output_heart": 1,
            "max_output": 5,
            "conversion_ticks": 1,
            "cooldown": 10,
            "initial_items": 1,
        },
        "mine_red": {
            "type_id": TYPE_MINE_RED,
            "output_ore.red": 1,
            "color": 0,
            "max_output": 5,
            "conversion_ticks": 1,
            "cooldown": 50,
            "initial_items": 1,
        },
        "generator_red": {
            "type_id": TYPE_GENERATOR_RED,
            "input_ore.red": 1,
            "output_battery.red": 1,
            "color": 0,
            "max_output": 5,
            "conversion_ticks": 1,
            "cooldown": 50,
            "initial_items": 1,
        },
        "wall": {"type_id": TYPE_WALL, "swappable": False},
        "block": {"type_id": TYPE_WALL, "swappable": True},
    }


def _get_default_actions():
    """Get default action configuration."""
    return {
        "noop": {"enabled": True},
        "move": {"enabled": True},
        "rotate": {"enabled": True},
        "put_items": {"enabled": True},
        "get_items": {"enabled": True},
        "attack": {"enabled": False},
        "swap": {"enabled": True},
        "change_color": {"enabled": False},
    }


def build_runtime_config(
    run: str = "default_run",
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 0,
    vectorization: str = "serial",
) -> Dict[str, Any]:
    """Build the runtime configuration for Metta."""
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "./train_dir")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "run": run,
        "data_dir": data_dir,
        "run_dir": f"{data_dir}/{run}",
        "policy_uri": f"file://{data_dir}/{run}/checkpoints",
        "torch_deterministic": True,
        "vectorization": vectorization,
        "seed": seed,
        "device": device,
        "stats_user": os.environ.get("USER", "unknown"),
        "dist_cfg_path": None,
        "hydra": {"callbacks": {"resolver_callback": {"_target_": "metta.common.util.resolvers.ResolverRegistrar"}}},
    }


def setup_metta_environment(config: Dict[str, Any]) -> None:
    """Setup the Metta environment with the given configuration."""
    from metta.common.util.runtime_configuration import setup_mettagrid_environment

    setup_mettagrid_environment(DictConfig(config))


def get_logger(name: str):
    """Get a configured logger for Metta."""
    from metta.common.util.logging import setup_mettagrid_logger

    return setup_mettagrid_logger(name)


# High-level convenience functions


def _setup_training_environment(
    run_name: str,
    num_agents: int,
    batch_size: int,
    num_workers: int,
    device: str,
    vectorization: str,
    env_width: int,
    env_height: int,
    logger,
) -> Tuple[Any, Any, torch.nn.Module, Any]:
    """Setup training environment, policy, and experience buffer."""
    import gymnasium as gym

    # Create environment
    env_config = env(
        num_agents=num_agents,
        width=env_width,
        height=env_height,
        max_steps=1000,
    )

    # Calculate environment count for batch size
    target_num_envs = batch_size // num_agents
    if target_num_envs < num_workers:
        target_num_envs = num_workers
        logger.warning(
            f"Requested batch_size {batch_size} with {num_agents} agents would need "
            f"{batch_size // num_agents} envs, but num_workers={num_workers}. "
            f"Using {target_num_envs} envs instead."
        )

    # Adjust to be multiple of num_workers
    num_envs = (target_num_envs // num_workers) * num_workers
    if num_envs == 0:
        num_envs = num_workers

    logger.info(f"Using {num_envs} environments to achieve batch size ~{num_envs * num_agents}")

    # Create vectorized environment
    vecenv = make_vecenv(
        env_config=env_config,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=num_envs // num_workers if num_workers > 1 else None,
        device=device,
        vectorization=vectorization,
    )

    env_info = vecenv.driver_env

    # Create observation space
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env_info.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Create agent
    device_obj = torch.device(device)
    policy = make_agent(
        obs_space=obs_space,
        action_space=env_info.single_action_space,
        obs_width=env_info.obs_width,
        obs_height=env_info.obs_height,
        feature_normalizations=env_info.feature_normalizations,
        global_features=env_info.global_features,
        device=device_obj,
    )
    policy.activate_actions(env_info.action_names, env_info.max_action_args, device_obj)

    # Create experience buffer
    total_agents = vecenv.num_agents
    experience = make_experience_buffer(
        total_agents=total_agents,
        batch_size=batch_size,
        obs_space=env_info.single_observation_space,
        atn_space=env_info.single_action_space,
        device=device_obj,
        hidden_size=policy.hidden_size,
    )

    return vecenv, env_info, policy, experience


def _train_epoch(
    policy,
    experience,
    loss_module,
    optimizer,
    losses,
    advantages,
    update_epochs: int,
    max_grad_norm: float,
    target_kl: Optional[float],
    grad_stats_interval: int,
    epoch: int,
    agent_step: int,
    device,
    logger,
) -> Tuple[bool, List[Tuple[int, Dict[str, float]]]]:
    """Train for one epoch, return early_stop flag and gradient stats."""
    kl_values = []
    gradient_stats_history = []
    early_stop = False

    minibatch_idx = 0
    total_minibatches = experience.num_minibatches * update_epochs

    for update_epoch in range(update_epochs):
        for mb_idx in range(experience.num_minibatches):
            minibatch = experience.sample_minibatch(
                advantages=advantages,
                prio_alpha=0.0,
                prio_beta=0.6,
                minibatch_idx=minibatch_idx,
                total_minibatches=total_minibatches,
            )

            loss = loss_module(
                minibatch=minibatch,
                experience=experience,
                losses=losses,
                agent_step=agent_step,
                device=device,
            )
            losses.minibatches_processed += 1

            optimizer.zero_grad()
            loss.backward()

            # Compute gradient statistics if enabled
            if grad_stats_interval > 0 and epoch % grad_stats_interval == 0:
                grad_stats = compute_gradient_stats(policy)
                gradient_stats_history.append((epoch, grad_stats))

            if (minibatch_idx + 1) % experience.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                if hasattr(policy, "clip_weights"):
                    policy.clip_weights()

            minibatch_idx += 1

            # Track KL for early stopping
            if hasattr(losses, "approx_kl_sum") and losses.minibatches_processed > 0:
                avg_kl = losses.approx_kl_sum / losses.minibatches_processed
                kl_values.append(avg_kl)

        # Check for early stopping based on KL divergence
        if target_kl is not None and kl_values:
            mean_kl = np.mean(kl_values)
            if mean_kl > target_kl:
                logger.info(f"Early stopping: KL divergence ({mean_kl:.4f}) exceeded target ({target_kl})")
                early_stop = True
                break

    return early_stop, gradient_stats_history


def _log_metrics(
    epoch: int,
    agent_step: int,
    losses,
    optimizer,
    timer,
    all_rollout_stats: Dict[str, Any],
    gradient_stats_history: List[Tuple[int, Dict[str, float]]],
    wandb_run,
    logger,
) -> float:
    """Log metrics and return mean reward."""
    loss_dict = losses.stats()

    # Log to wandb if enabled
    if wandb_run:
        try:
            import wandb

            metrics = {
                "epoch": epoch,
                "agent_step": agent_step,
                "loss/policy": loss_dict.get("policy_loss", 0),
                "loss/value": loss_dict.get("value_loss", 0),
                "loss/entropy": loss_dict.get("entropy", 0),
                "metrics/approx_kl": loss_dict.get("approx_kl", 0),
                "metrics/clipfrac": loss_dict.get("clipfrac", 0),
                "metrics/explained_variance": loss_dict.get("explained_variance", 0),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }

            # Add rollout stats
            if all_rollout_stats:
                for key, values in all_rollout_stats.items():
                    if isinstance(values, list) and values:
                        metrics[f"rollout/{key}"] = np.mean(values)

            # Add gradient stats
            if gradient_stats_history and gradient_stats_history[-1][0] == epoch:
                for key, value in gradient_stats_history[-1][1].items():
                    metrics[f"gradients/{key}"] = value

            wandb.log(metrics, step=agent_step)
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    # Calculate timing metrics
    rollout_time = timer.get_last_elapsed("rollout")
    train_time = timer.get_last_elapsed("train")
    stats_time = timer.get_last_elapsed("stats")
    total_time = rollout_time + train_time + stats_time

    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0
    stats_pct = (stats_time / total_time) * 100 if total_time > 0 else 0

    # Calculate average reward
    mean_reward = 0.0
    if all_rollout_stats:
        reward_keys = [k for k in all_rollout_stats.keys() if "reward" in k and "mean" in k]
        if reward_keys:
            all_rewards = []
            for k in reward_keys:
                if isinstance(all_rollout_stats[k], list):
                    all_rewards.extend(all_rollout_stats[k])
                else:
                    all_rewards.append(all_rollout_stats[k])
            if all_rewards:
                mean_reward = np.mean(all_rewards)

    # Log training progress
    logger.info(
        f"Epoch {epoch} - "
        f"{agent_step / total_time:.0f} steps/sec "
        f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats) - "
        f"reward: {mean_reward:.2f}, "
        f"policy_loss: {loss_dict.get('policy_loss', 0):.4f}, "
        f"value_loss: {loss_dict.get('value_loss', 0):.4f}, "
        f"kl: {loss_dict.get('approx_kl', 0):.4f}"
    )

    return mean_reward


def quick_train(
    run_name: str = "default_run",
    timesteps: int = 50_000_000_000,
    batch_size: int = None,
    num_agents: int = None,
    num_workers: int = 1,
    learning_rate: float = None,
    checkpoint_interval: int = 60,
    evaluate_interval: int = 300,
    device: str = "cuda",
    vectorization: str = "serial",
    env_width: int = None,
    env_height: int = None,
    bptt_horizon: int = None,
    minibatch_size: int = None,
    update_epochs: int = None,
    max_grad_norm: float = None,
    target_kl: Optional[float] = None,
    anneal_lr: bool = False,
    lr_schedule_type: str = "linear",
    warmup_steps: Optional[int] = None,
    l2_init_weight_update_interval: int = 0,
    grad_stats_interval: int = 0,
    save_full_state: bool = True,
    wandb_enabled: bool = False,
    wandb_project: str = "metta",
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    resume_from: Optional[str] = None,
    logger=None,
    **kwargs,
) -> str:
    """Quick training function with sensible defaults.

    Returns:
        Path to the final checkpoint
    """
    import os
    import time

    from metta.common.stopwatch import Stopwatch
    from metta.rl.functions import (
        compute_advantage,
        perform_rollout_step,
        process_rollout_infos,
    )
    from metta.rl.losses import Losses

    # Apply defaults
    params = {**DEFAULT_PARAMS, **kwargs}
    batch_size = batch_size or params["batch_size"]
    num_agents = num_agents or DEFAULT_ENV_PARAMS["num_agents"]
    learning_rate = learning_rate or params["learning_rate"]
    env_width = env_width or DEFAULT_ENV_PARAMS["width"]
    env_height = env_height or DEFAULT_ENV_PARAMS["height"]
    bptt_horizon = bptt_horizon or params["bptt_horizon"]
    minibatch_size = minibatch_size or params["minibatch_size"]
    update_epochs = update_epochs or params["update_epochs"]
    max_grad_norm = max_grad_norm or params["max_grad_norm"]

    if logger is None:
        logger = get_logger("quick_train")

    # Initialize wandb if enabled
    wandb_run = None
    if wandb_enabled:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                tags=wandb_tags,
                name=run_name,
                config={
                    "run_name": run_name,
                    "timesteps": timesteps,
                    "batch_size": batch_size,
                    "num_agents": num_agents,
                    "learning_rate": learning_rate,
                    "device": device,
                    **params,
                },
            )
            logger.info(f"Initialized wandb run: {wandb_run.name}")
        except ImportError:
            logger.warning("Wandb enabled but not installed. Install with: pip install wandb")
            wandb_enabled = False

    # Setup directories
    checkpoint_dir = f"./train_dir/{run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup training environment
    vecenv, env_info, policy, experience = _setup_training_environment(
        run_name=run_name,
        num_agents=num_agents,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        vectorization=vectorization,
        env_width=env_width,
        env_height=env_height,
        logger=logger,
    )

    # Create optimizer and loss module
    optimizer = make_optimizer(policy.parameters(), learning_rate=learning_rate)
    loss_module = make_loss_module(policy=policy, **params)
    losses = Losses()

    # Create learning rate scheduler
    lr_scheduler = None
    if anneal_lr:
        lr_scheduler = make_lr_scheduler(
            optimizer=optimizer,
            total_timesteps=timesteps,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            schedule_type=lr_schedule_type,
            anneal_lr=anneal_lr,
        )

    # Initialize training variables
    epoch = 0
    agent_step = 0
    latest_policy_path = None
    early_stop = False
    timer = Stopwatch(logger)
    start_time = time.time()
    all_rollout_stats = {}
    gradient_stats_history = []

    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        training_state = TrainingState.load(resume_from)
        epoch = training_state.epoch
        agent_step = training_state.agent_step
        optimizer.load_state_dict(training_state.optimizer_state_dict)
        if lr_scheduler and training_state.lr_scheduler_state_dict:
            lr_scheduler.load_state_dict(training_state.lr_scheduler_state_dict)
        if training_state.policy_path:
            policy.load_state_dict(torch.load(training_state.policy_path, map_location=device))
        if training_state.stopwatch_state:
            timer.load_state(training_state.stopwatch_state)
        logger.info(f"Resumed from epoch {epoch}, agent_step {agent_step}")

    logger.info(
        f"Starting training with {vecenv.num_envs} environments, "
        f"{vecenv.num_agents} total agents, "
        f"batch size {batch_size}, "
        f"minibatch size {experience.minibatch_size}"
    )

    # Reset environments
    vecenv.async_reset(seed=0)
    device_obj = torch.device(device)

    # Main training loop
    while agent_step < timesteps and not early_stop:
        steps_before = agent_step

        # Rollout phase
        with timer("rollout"):
            raw_infos = []
            experience.reset_for_rollout()

            while not experience.ready_for_training:
                num_steps, info, _ = perform_rollout_step(
                    policy=policy,
                    vecenv=vecenv,
                    experience=experience,
                    device=device_obj,
                    timer=timer,
                )
                agent_step += num_steps
                if info:
                    raw_infos.extend(info)

            # Process rollout stats
            rollout_stats = process_rollout_infos(raw_infos)
            for k, v in rollout_stats.items():
                if k not in all_rollout_stats:
                    all_rollout_stats[k] = []
                if isinstance(v, list):
                    all_rollout_stats[k].extend(v)
                else:
                    all_rollout_stats[k].append(v)

        # Train phase
        with timer("train"):
            losses.zero()
            experience.reset_importance_sampling_ratios()

            # Compute advantages
            advantages = torch.zeros(experience.values.shape, device=device_obj)
            initial_importance_sampling_ratio = torch.ones_like(experience.values)
            advantages = compute_advantage(
                experience.values,
                experience.rewards,
                experience.dones,
                initial_importance_sampling_ratio,
                advantages,
                params["gamma"],
                params["gae_lambda"],
                params["vtrace_rho_clip"],
                params["vtrace_c_clip"],
                device_obj,
            )

            # Train for one epoch
            early_stop, new_grad_stats = _train_epoch(
                policy=policy,
                experience=experience,
                loss_module=loss_module,
                optimizer=optimizer,
                losses=losses,
                advantages=advantages,
                update_epochs=update_epochs,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                grad_stats_interval=grad_stats_interval,
                epoch=epoch,
                agent_step=agent_step,
                device=device_obj,
                logger=logger,
            )
            gradient_stats_history.extend(new_grad_stats)
            epoch += update_epochs

            # Update learning rate scheduler
            if lr_scheduler:
                lr_scheduler.step()

            # Update L2 init weights if enabled
            if l2_init_weight_update_interval > 0 and epoch % l2_init_weight_update_interval == 0:
                if hasattr(policy, "update_l2_init_weight_copy"):
                    policy.update_l2_init_weight_copy()
                    logger.info(f"Updated L2 init weights at epoch {epoch}")

        # Process stats
        with timer("stats"):
            mean_reward = _log_metrics(
                epoch=epoch,
                agent_step=agent_step,
                losses=losses,
                optimizer=optimizer,
                timer=timer,
                all_rollout_stats=all_rollout_stats,
                gradient_stats_history=gradient_stats_history,
                wandb_run=wandb_run,
                logger=logger,
            )

        # Epoch-based checkpoint saving
        if epoch % checkpoint_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/policy_epoch_{epoch}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            latest_policy_path = checkpoint_path
            logger.info(f"Saved policy checkpoint: {checkpoint_path}")

            # Save full training state if enabled
            if save_full_state:
                training_state = TrainingState(
                    epoch=epoch,
                    agent_step=agent_step,
                    total_agent_step=agent_step,
                    optimizer_state_dict=optimizer.state_dict(),
                    lr_scheduler_state_dict=lr_scheduler.state_dict() if lr_scheduler else None,
                    policy_path=checkpoint_path,
                    stopwatch_state=timer.save_state(),
                    extra_args={"run_name": run_name, "mean_reward": mean_reward},
                )
                state_path = training_state.save(checkpoint_dir)
                logger.info(f"Saved full training state: {state_path}")

        # Periodic evaluation if enabled
        if evaluate_interval > 0 and epoch % evaluate_interval == 0:
            logger.info(f"Running evaluation at epoch {epoch}")
            eval_results = quick_eval(
                checkpoint_path=latest_policy_path or f"{checkpoint_dir}/policy_epoch_{epoch}.pt",
                num_episodes=10,
                num_envs=min(32, vecenv.num_envs),
                num_agents=num_agents,
                device=device,
                logger=logger,
            )
            logger.info(
                f"Evaluation results: avg_reward={eval_results['avg_reward']:.2f}, "
                f"episodes={eval_results['num_episodes']}"
            )

            # Log eval results to wandb
            if wandb_enabled and wandb_run:
                try:
                    import wandb

                    wandb.log(
                        {
                            "eval/avg_reward": eval_results["avg_reward"],
                            "eval/std_reward": eval_results["std_reward"],
                            "eval/min_reward": eval_results["min_reward"],
                            "eval/max_reward": eval_results["max_reward"],
                        },
                        step=agent_step,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log eval results to wandb: {e}")

        # Clear accumulated stats periodically
        if epoch % 10 == 0:
            all_rollout_stats.clear()

    # Save final checkpoint
    final_checkpoint = f"{checkpoint_dir}/policy_final.pt"
    torch.save(policy.state_dict(), final_checkpoint)
    logger.info(f"Training complete! Final checkpoint: {final_checkpoint}")

    # Save final training state
    if save_full_state:
        final_state = TrainingState(
            epoch=epoch,
            agent_step=agent_step,
            total_agent_step=agent_step,
            optimizer_state_dict=optimizer.state_dict(),
            lr_scheduler_state_dict=lr_scheduler.state_dict() if lr_scheduler else None,
            policy_path=final_checkpoint,
            stopwatch_state=timer.save_state(),
            extra_args={"run_name": run_name, "final": True},
        )
        final_state_path = final_state.save(checkpoint_dir)
        logger.info(f"Saved final training state: {final_state_path}")

    # Log timing summary
    elapsed_time = time.time() - start_time
    logger.info(f"Total training time: {elapsed_time:.1f}s")
    logger.info(f"Average SPS: {agent_step / elapsed_time:.0f}")

    # Log gradient stats summary
    if gradient_stats_history:
        logger.info("Gradient statistics summary:")
        recent_stats = gradient_stats_history[-10:]  # Last 10 measurements
        for stat_name in ["grad_norm_mean", "grad_norm_max", "param_norm_mean"]:
            values = [s[1].get(stat_name, 0) for s in recent_stats if stat_name in s[1]]
            if values:
                logger.info(f"  {stat_name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    # Close wandb run
    if wandb_enabled and wandb_run:
        try:
            import wandb

            wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to close wandb run: {e}")

    vecenv.close()
    return final_checkpoint


def quick_eval(
    checkpoint_path: str,
    num_episodes: int = 10,
    num_envs: int = 32,
    num_agents: int = None,
    device: str = "cuda",
    vectorization: str = "multiprocessing",
    env_width: int = None,
    env_height: int = None,
    logger=None,
    **kwargs,
) -> Dict[str, Any]:
    """Quick evaluation function.

    Returns:
        Dictionary with evaluation results
    """
    import gymnasium as gym

    from metta.agent.policy_state import PolicyState
    from metta.mettagrid.mettagrid_env import dtype_actions
    from metta.mettagrid.util.dict_utils import unroll_nested_dict

    # Apply defaults
    num_agents = num_agents or DEFAULT_ENV_PARAMS["num_agents"]
    env_width = env_width or DEFAULT_ENV_PARAMS["width"]
    env_height = env_height or DEFAULT_ENV_PARAMS["height"]

    if logger is None:
        logger = get_logger("quick_eval")

    # Create environment
    env_config = env(num_agents=num_agents, width=env_width, height=env_height, **kwargs)
    vecenv = make_vecenv(
        env_config=env_config,
        num_envs=num_envs,
        num_workers=1,
        device=device,
        vectorization=vectorization,
    )

    env_info = vecenv.driver_env
    device_obj = torch.device(device)

    # Create observation space and agent
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env_info.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    policy = make_agent(
        obs_space=obs_space,
        action_space=env_info.single_action_space,
        obs_width=env_info.obs_width,
        obs_height=env_info.obs_height,
        feature_normalizations=env_info.feature_normalizations,
        global_features=env_info.global_features,
        device=device_obj,
    )
    policy.activate_actions(env_info.action_names, env_info.max_action_args, device_obj)

    # Load checkpoint and set to eval mode
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # Initialize state
    state = PolicyState()
    if hasattr(policy, "core_num_layers"):
        state.lstm_h = torch.zeros(policy.core_num_layers, vecenv.num_agents, policy.hidden_size, device=device_obj)
        state.lstm_c = torch.zeros(policy.core_num_layers, vecenv.num_agents, policy.hidden_size, device=device_obj)

    # Run evaluation
    rewards = []
    episode_lengths = []
    episodes_completed = 0

    vecenv.async_reset(seed=42)
    logger.info(f"Starting evaluation with {num_envs} environments, collecting {num_episodes} episodes")

    while episodes_completed < num_episodes:
        # Receive and process
        o, r, d, t, info, env_id, mask = vecenv.recv()
        o = torch.as_tensor(o).to(device_obj, non_blocking=True)

        with torch.no_grad():
            actions, _, _, _, _ = policy(o, state)

        vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Process episode completions
        if info:
            for info_dict in info:
                if info_dict:
                    flat_info = dict(unroll_nested_dict(info_dict))

                    # Look for episode completion
                    for key, value in flat_info.items():
                        if key.startswith("task_reward/") and key.endswith("/rewards.mean"):
                            rewards.append(float(value))

                            # Get episode length if available
                            if "attributes" in flat_info and isinstance(flat_info["attributes"], dict):
                                if "steps" in flat_info["attributes"]:
                                    episode_lengths.append(int(flat_info["attributes"]["steps"]))

                            episodes_completed += 1
                            if episodes_completed % max(1, num_episodes // 10) == 0:
                                logger.info(f"Episodes completed: {episodes_completed}/{num_episodes}")

                            if episodes_completed >= num_episodes:
                                break

                if episodes_completed >= num_episodes:
                    break

    vecenv.close()

    # Compute results
    results = {
        "num_episodes": len(rewards),
        "avg_reward": np.mean(rewards) if rewards else 0.0,
        "std_reward": np.std(rewards) if rewards else 0.0,
        "min_reward": np.min(rewards) if rewards else 0.0,
        "max_reward": np.max(rewards) if rewards else 0.0,
    }

    if episode_lengths:
        results["avg_episode_length"] = np.mean(episode_lengths)

    return results


def quick_sim(
    run_name: str,
    policy_uri: str,
    num_episodes: int = 10,
    num_envs: int = 32,
    num_agents: int = None,
    device: str = "cuda",
    logger=None,
    **kwargs,
) -> Dict[str, Any]:
    """Quick simulation/evaluation function using direct evaluation.

    Returns:
        Dictionary with simulation results
    """
    import os

    if logger is None:
        logger = get_logger("quick_sim")

    # Extract checkpoint path from URI
    checkpoint_path = policy_uri[7:] if policy_uri.startswith("file://") else policy_uri
    checkpoint_path = os.path.abspath(checkpoint_path)

    logger.info(f"Evaluating policy: {checkpoint_path}")

    # Use quick_eval to run the evaluation
    results = quick_eval(
        checkpoint_path=checkpoint_path,
        num_episodes=num_episodes,
        num_envs=num_envs,
        num_agents=num_agents,
        device=device,
        vectorization="multiprocessing",
        logger=logger,
        **kwargs,
    )

    # Format results
    return {
        "policies": [
            {
                "name": os.path.basename(checkpoint_path),
                "uri": policy_uri,
                "metrics": results,
            }
        ]
    }


# Advanced features for production use


def create_policy_store(config: Dict[str, Any]) -> Any:
    """Create a PolicyStore instance for managing policies."""
    from metta.agent.policy_store import PolicyStore

    return PolicyStore(DictConfig(config), stats_client=None)


def save_policy_to_store(
    policy_store: Any,
    policy: torch.nn.Module,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Save a policy to the PolicyStore with metadata."""
    import os

    # Create path for policy
    path = os.path.join(policy_store.policy_uri.replace("file://", ""), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save policy with metadata
    return policy_store.save(name, path, policy, metadata or {})


def run_simulation_suite(
    policy_path: str,
    suite_name: str = "eval",
    num_envs: int = 32,
    num_episodes: int = 10,
    device: str = "cuda",
    logger=None,
) -> Dict[str, Any]:
    """Run a full simulation suite evaluation."""
    import os

    from metta.agent.policy_record import PolicyRecord
    from metta.sim.simulation import SimulationSuite
    from metta.sim.simulation_config import SimulationSuiteConfig

    if logger is None:
        logger = get_logger("simulation_suite")

    # Create simulation suite config
    suite_config = SimulationSuiteConfig(
        name=suite_name,
        num_envs=num_envs,
        num_episodes=num_episodes,
        map_preview_limit=32,
        suites=[],  # Will be populated based on suite_name
    )

    # Create policy record
    policy_record = PolicyRecord(
        name=os.path.basename(policy_path),
        uri=f"file://{os.path.abspath(policy_path)}",
        generation=1,
    )

    # Run simulation suite
    logger.info(f"Running simulation suite '{suite_name}' with policy: {policy_path}")
    sim_suite = SimulationSuite(config=suite_config, policy_pr=policy_record, device=device)
    results = sim_suite.run()

    # Format results
    formatted_results = {}
    for task_name, task_results in results.items():
        if isinstance(task_results, dict) and "metrics" in task_results:
            formatted_results[task_name] = {
                "avg_reward": task_results["metrics"].get("mean", 0),
                "std_reward": task_results["metrics"].get("std", 0),
                "episodes": task_results["metrics"].get("count", 0),
            }

    return formatted_results


def generate_replay(
    policy_path: str,
    num_episodes: int = 1,
    output_dir: str = "./replays",
    device: str = "cuda",
    logger=None,
) -> List[str]:
    """Generate replay files for visualization.

    Note: This is a placeholder - actual replay generation would require
    integration with the replay recording system.
    """
    if logger is None:
        logger = get_logger("replay_generator")

    logger.info("Replay generation not yet implemented in functional API")
    return []
