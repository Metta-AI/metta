"""
Clean API for Metta - provides direct instantiation without Hydra.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
        "components": {
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
        },
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


def make_optimizer(
    parameters,
    learning_rate: float = 0.0004573146765703167,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-12,
    weight_decay: float = 0,
    type: str = "adam",
) -> torch.optim.Optimizer:
    """Create an optimizer directly."""
    if type == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {type}")


def make_experience_buffer(
    total_agents: int,
    batch_size: int,
    bptt_horizon: int,
    minibatch_size: int,
    max_minibatch_size: int,
    obs_space,
    atn_space,
    device: torch.device,
    hidden_size: int,
    cpu_offload: bool = False,
    num_lstm_layers: int = 2,
    agents_per_batch: Optional[int] = None,
):
    """Create an experience buffer directly."""
    from metta.rl.experience import Experience

    return Experience(
        total_agents=total_agents,
        batch_size=batch_size,
        bptt_horizon=bptt_horizon,
        minibatch_size=minibatch_size,
        max_minibatch_size=max_minibatch_size,
        obs_space=obs_space,
        atn_space=atn_space,
        device=device,
        hidden_size=hidden_size,
        cpu_offload=cpu_offload,
        num_lstm_layers=num_lstm_layers,
        agents_per_batch=agents_per_batch,
    )


def make_loss_module(
    policy: torch.nn.Module,
    vf_coef: float = 0.44,
    ent_coef: float = 0.0021,
    clip_coef: float = 0.1,
    vf_clip_coef: float = 0.1,
    norm_adv: bool = True,
    clip_vloss: bool = True,
    gamma: float = 0.977,
    gae_lambda: float = 0.916,
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
    l2_reg_loss_coef: float = 0.0,
    l2_init_loss_coef: float = 0.0,
    kickstarter: Optional[Any] = None,
):
    """Create a PPO loss module directly."""
    from metta.rl.objectives import ClipPPOLoss

    return ClipPPOLoss(
        policy=policy,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        clip_coef=clip_coef,
        vf_clip_coef=vf_clip_coef,
        norm_adv=norm_adv,
        clip_vloss=clip_vloss,
        gamma=gamma,
        gae_lambda=gae_lambda,
        vtrace_rho_clip=vtrace_rho_clip,
        vtrace_c_clip=vtrace_c_clip,
        l2_reg_loss_coef=l2_reg_loss_coef,
        l2_init_loss_coef=l2_init_loss_coef,
        kickstarter=kickstarter,
    )


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


def env(
    num_agents: int = 2,
    width: int = 15,
    height: int = 10,
    max_steps: int = 1000,
    obs_width: int = 11,
    obs_height: int = 11,
) -> Dict[str, Any]:
    """Create a default MetaGrid environment configuration."""
    return {
        "sampling": 0,
        "desync_episodes": False,
        "replay_level_prob": 0.0,
        "game": {
            "num_agents": num_agents,
            "obs_width": obs_width,
            "obs_height": obs_height,
            "num_observation_tokens": 200,
            "max_steps": max_steps,
            "inventory_item_names": [
                "ore.red",
                "battery.red",
                "heart",
                "laser",
                "armor",
            ],
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
            "objects": {
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
                "wall": {
                    "type_id": TYPE_WALL,
                    "swappable": False,
                },
                "block": {
                    "type_id": TYPE_WALL,
                    "swappable": True,
                },
            },
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "attack": {"enabled": False},
                "swap": {"enabled": True},
                "change_color": {"enabled": False},
            },
            "reward_sharing": {"groups": {}},
            "map_builder": {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": width,
                "height": height,
                "border_width": 2,
                "agents": num_agents,
                "objects": {"mine_red": 2, "generator_red": 1, "altar": 1, "wall": 5, "block": 3},
            },
        },
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


def quick_train(
    run_name: str = "default_run",
    timesteps: int = 50_000_000_000,  # Match original default
    batch_size: int = 262_144,  # Match original default
    num_agents: int = 2,
    num_workers: int = 1,
    learning_rate: float = 0.0004573146765703167,  # Match original default
    checkpoint_interval: int = 60,  # epochs, not seconds
    evaluate_interval: int = 300,  # epochs
    device: str = "cuda",
    vectorization: str = "serial",
    env_width: int = 15,
    env_height: int = 10,
    bptt_horizon: int = 64,  # Match original default
    minibatch_size: int = 16_384,  # Match original default
    update_epochs: int = 1,  # Match original default
    max_grad_norm: float = 0.5,  # Match original default
    # New parameters for enhanced features
    target_kl: Optional[float] = None,  # Early stopping based on KL divergence
    anneal_lr: bool = False,  # Enable learning rate annealing
    lr_schedule_type: str = "linear",  # Type of LR schedule
    warmup_steps: Optional[int] = None,  # Warmup steps for LR scheduler
    l2_init_weight_update_interval: int = 0,  # L2 weight update interval
    grad_stats_interval: int = 0,  # Gradient statistics logging interval
    save_full_state: bool = True,  # Save full training state
    wandb_enabled: bool = False,  # Enable wandb logging
    wandb_project: str = "metta",  # Wandb project name
    wandb_entity: Optional[str] = None,  # Wandb entity
    wandb_tags: Optional[List[str]] = None,  # Wandb tags
    resume_from: Optional[str] = None,  # Resume from checkpoint
    logger=None,
) -> str:
    """Quick training function with sensible defaults matching the original trainer.

    Args:
        run_name: Name of the training run
        timesteps: Total timesteps to train
        batch_size: Batch size for training
        num_agents: Number of agents per environment
        num_workers: Number of workers
        learning_rate: Learning rate
        checkpoint_interval: How often to save checkpoints (in epochs)
        evaluate_interval: How often to evaluate (in epochs)
        device: Device to use
        vectorization: Vectorization mode
        env_width: Environment width
        env_height: Environment height
        bptt_horizon: BPTT horizon for LSTM training
        minibatch_size: Minibatch size
        update_epochs: Number of epochs to update per rollout
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Stop training if KL divergence exceeds this value
        anneal_lr: Whether to anneal learning rate
        lr_schedule_type: Type of LR schedule ("linear" or "cosine")
        warmup_steps: Number of warmup steps for LR scheduler
        l2_init_weight_update_interval: How often to update L2 init weights (epochs)
        grad_stats_interval: How often to compute gradient statistics (epochs)
        save_full_state: Whether to save full training state (optimizer, etc.)
        wandb_enabled: Whether to enable wandb logging
        wandb_project: Wandb project name
        wandb_entity: Wandb entity
        wandb_tags: Wandb tags
        resume_from: Path to checkpoint to resume from
        logger: Optional logger instance

    Returns:
        Path to the final checkpoint
    """
    import os
    import time

    import gymnasium as gym

    from metta.common.stopwatch import Stopwatch
    from metta.rl.functional_trainer import (
        compute_initial_advantages,
        perform_rollout_step,
        process_rollout_infos,
    )
    from metta.rl.losses import Losses

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
                    "bptt_horizon": bptt_horizon,
                    "minibatch_size": minibatch_size,
                    "update_epochs": update_epochs,
                    "target_kl": target_kl,
                    "anneal_lr": anneal_lr,
                },
            )
            logger.info(f"Initialized wandb run: {wandb_run.name}")
        except ImportError:
            logger.warning("Wandb enabled but not installed. Install with: pip install wandb")
            wandb_enabled = False

    # Setup directories
    checkpoint_dir = f"./train_dir/{run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create environment
    env_config = env(
        num_agents=num_agents,
        width=env_width,
        height=env_height,
        max_steps=1000,
    )

    # Calculate environment count for batch size
    # The number of environments should be calculated to achieve desired batch size
    # batch_size = num_envs * num_agents * steps_per_rollout
    # For single-step rollouts, we need: num_envs = batch_size / num_agents
    target_num_envs = batch_size // num_agents

    # Ensure we have at least as many envs as workers
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
    logger.info(f"Total agents: {num_envs * num_agents}")

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

    # For experience buffer, use actual batch_size parameter
    # The original trainer uses a different batch_size for experience vs environments
    total_agents = vecenv.num_agents

    # Validate batch_size is large enough
    min_batch_size = total_agents * bptt_horizon
    if batch_size < min_batch_size:
        logger.warning(
            f"Batch size {batch_size} is too small for {total_agents} agents with "
            f"bptt_horizon {bptt_horizon}. Adjusting to minimum {min_batch_size}."
        )
        batch_size = min_batch_size

    # Create experience buffer with proper minibatch calculation
    # Ensure minibatch_size divides batch_size evenly
    while batch_size % minibatch_size != 0 and minibatch_size > 1:
        minibatch_size -= 1

    experience = make_experience_buffer(
        total_agents=total_agents,
        batch_size=batch_size,
        bptt_horizon=bptt_horizon,
        minibatch_size=minibatch_size,
        max_minibatch_size=minibatch_size,
        obs_space=env_info.single_observation_space,
        atn_space=env_info.single_action_space,
        device=device_obj,
        hidden_size=policy.hidden_size,
        num_lstm_layers=policy.core_num_layers,
        agents_per_batch=getattr(vecenv, "agents_per_batch", None),
    )

    # Create optimizer and loss module
    optimizer = make_optimizer(policy.parameters(), learning_rate=learning_rate)
    loss_module = make_loss_module(policy=policy)
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
    steps_per_epoch = batch_size  # Steps taken per rollout
    total_epochs = timesteps // batch_size  # Total number of rollout+train cycles
    latest_policy_path = None
    early_stop = False

    # Timing
    timer = Stopwatch(logger)
    start_time = time.time()

    # For distributed training (currently single GPU)
    world_size = 1

    # Stats
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
        f"Starting training with {num_envs} environments, "
        f"{total_agents} total agents, "
        f"batch size {batch_size}, "
        f"minibatch size {experience.minibatch_size}"
    )

    # Reset environments
    vecenv.async_reset(seed=0)

    # Main training loop - matches original trainer structure
    while agent_step < timesteps and not early_stop:
        steps_before = agent_step

        # Rollout phase
        with timer("rollout"):
            raw_infos = []

            # Reset experience buffer for new rollout
            experience.reset_for_rollout()

            while not experience.ready_for_training:
                # Rollout single step
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

            # Accumulate stats
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
            advantages = compute_initial_advantages(
                experience, gamma=0.977, gae_lambda=0.916, vtrace_rho_clip=1.0, vtrace_c_clip=1.0, device=device_obj
            )

            # Track KL divergence for early stopping
            kl_values = []

            # Update epochs (inner loop)
            minibatch_idx = 0
            total_minibatches = experience.num_minibatches * update_epochs

            for update_epoch in range(update_epochs):
                # Train minibatches
                for mb_idx in range(experience.num_minibatches):
                    minibatch = experience.sample_minibatch(
                        advantages=advantages,
                        prio_alpha=0.0,  # No prioritized replay by default
                        prio_beta=0.6,
                        minibatch_idx=minibatch_idx,
                        total_minibatches=total_minibatches,
                    )

                    loss = loss_module(
                        minibatch=minibatch,
                        experience=experience,
                        losses=losses,
                        agent_step=agent_step,
                        device=device_obj,
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

                # Increment epoch after all minibatches in update_epoch
                epoch += 1

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
            loss_dict = losses.stats()

            # Log to wandb if enabled
            if wandb_enabled and wandb_run:
                try:
                    import wandb

                    # Prepare metrics
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

        # Calculate and log metrics (per rollout+train cycle, not per epoch)
        steps_in_cycle = agent_step - steps_before
        rollout_time = timer.get_last_elapsed("rollout")
        train_time = timer.get_last_elapsed("train")
        stats_time = timer.get_last_elapsed("stats")
        total_time = rollout_time + train_time + stats_time
        steps_per_sec = steps_in_cycle / total_time if total_time > 0 else 0

        train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
        rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0
        stats_pct = (stats_time / total_time) * 100 if total_time > 0 else 0

        # Calculate average reward from stats
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

        loss_stats = losses.stats()

        # Enhanced logging with loss stats
        logger.info(
            f"Epoch {epoch} - "
            f"{steps_per_sec * world_size:.0f} steps/sec "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats) - "
            f"reward: {mean_reward:.2f}, "
            f"policy_loss: {loss_stats.get('policy_loss', 0):.4f}, "
            f"value_loss: {loss_stats.get('value_loss', 0):.4f}, "
            f"kl: {loss_stats.get('approx_kl', 0):.4f}"
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
                    total_agent_step=agent_step * world_size,
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
                num_envs=min(32, num_envs),
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
            total_agent_step=agent_step * world_size,
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
    num_agents: int = 2,
    device: str = "cuda",
    vectorization: str = "multiprocessing",
    env_width: int = 15,
    env_height: int = 10,
    logger=None,
) -> Dict[str, Any]:
    """Quick evaluation function.

    Args:
        checkpoint_path: Path to checkpoint to evaluate
        num_episodes: Number of episodes to run
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        device: Device to use
        vectorization: Vectorization mode
        env_width: Environment width
        env_height: Environment height
        logger: Optional logger instance

    Returns:
        Dictionary with evaluation results
    """
    import gymnasium as gym

    if logger is None:
        logger = get_logger("quick_eval")

    # Create environment
    env_config = env(
        num_agents=num_agents,
        width=env_width,
        height=env_height,
        max_steps=1000,
    )

    # Create vectorized environment
    vecenv = make_vecenv(
        env_config=env_config,
        num_envs=num_envs,
        num_workers=1,
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

    # Load checkpoint
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()

    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # Run evaluation
    rewards = []
    episode_lengths = []
    episodes_completed = 0

    # Reset and start environments
    vecenv.async_reset(seed=42)

    # Initialize hidden state
    from metta.agent.policy_state import PolicyState
    from metta.mettagrid.mettagrid_env import dtype_actions

    state = PolicyState()
    if hasattr(policy, "core_num_layers"):
        state.lstm_h = torch.zeros(policy.core_num_layers, vecenv.num_agents, policy.hidden_size, device=device_obj)
        state.lstm_c = torch.zeros(policy.core_num_layers, vecenv.num_agents, policy.hidden_size, device=device_obj)

    step_count = 0

    logger.info(f"Starting evaluation with {num_envs} environments, collecting {num_episodes} episodes")

    while episodes_completed < num_episodes:
        # Receive from environment
        o, r, d, t, info, env_id, mask = vecenv.recv()
        step_count += 1

        # Convert observations to tensors
        o = torch.as_tensor(o).to(device_obj, non_blocking=True)

        with torch.no_grad():
            actions, _, _, _, _ = policy(o, state)

        # Send actions to environment
        vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Process episode completions
        if info:
            # Debug first few steps to see info structure
            if step_count <= 5:
                logger.info(f"Info at step {step_count}: {info}")

            # Process info like in training - it might be nested
            from metta.mettagrid.util.dict_utils import unroll_nested_dict

            for idx, info_dict in enumerate(info):
                if info_dict:
                    # Unroll nested dictionary
                    flat_info = dict(unroll_nested_dict(info_dict))

                    # Check various possible keys for episode completion
                    episode_done = False
                    episode_return = None
                    episode_length = None

                    # Look for task_reward pattern (e.g., "task_reward/task/rewards.mean")
                    for key, value in flat_info.items():
                        if key.startswith("task_reward/") and key.endswith("/rewards.mean"):
                            episode_return = value
                            episode_done = True
                            logger.info(f"Found episode completion with key: {key} = {value}")
                            break

                    # Also check for episode length/steps
                    if "attributes" in flat_info and isinstance(flat_info["attributes"], dict):
                        if "steps" in flat_info["attributes"]:
                            episode_length = flat_info["attributes"]["steps"]

                    if episode_done and episode_return is not None:
                        rewards.append(float(episode_return))
                        if episode_length is not None:
                            episode_lengths.append(int(episode_length))
                        episodes_completed += 1

                        logger.info(
                            f"Episode {episodes_completed}/{num_episodes} completed: "
                            f"reward={episode_return:.2f}, length={episode_length or 'N/A'}"
                        )

                        if episodes_completed >= num_episodes:
                            logger.info(f"Collected {num_episodes} episodes after {step_count} steps")
                            break

        # Debug: check what's in info periodically
        if step_count % 5000 == 0 and info:
            logger.info(f"Sample info at step {step_count}: {info[0] if info else 'None'}")

        # Log progress every 1000 steps
        if step_count % 1000 == 0:
            logger.info(f"Evaluation step {step_count}, episodes completed: {episodes_completed}/{num_episodes}")

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
        results["episode_lengths"] = episode_lengths

    return results


def quick_sim(
    run_name: str,
    policy_uri: str,
    num_episodes: int = 10,
    num_envs: int = 32,
    num_agents: int = 2,
    device: str = "cuda",
    logger=None,
) -> Dict[str, Any]:
    """Quick simulation/evaluation function using direct evaluation.

    Args:
        run_name: Name of the run
        policy_uri: URI of the policy to evaluate
        num_episodes: Number of episodes to run
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        device: Device to use
        logger: Optional logger instance

    Returns:
        Dictionary with simulation results
    """
    import os

    if logger is None:
        logger = get_logger("quick_sim")

    # Extract checkpoint path from URI
    if policy_uri.startswith("file://"):
        checkpoint_path = policy_uri[7:]
    else:
        checkpoint_path = policy_uri

    # Make sure path is absolute
    if not os.path.isabs(checkpoint_path):
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
    )

    # Format results
    policy_name = os.path.basename(checkpoint_path)
    return {
        "policies": [
            {
                "name": policy_name,
                "uri": policy_uri,
                "metrics": results,
            }
        ]
    }


# Advanced features for production use


def create_policy_store(config: Dict[str, Any]) -> Any:
    """Create a PolicyStore instance for managing policies.

    Args:
        config: Runtime configuration dict

    Returns:
        PolicyStore instance
    """
    from metta.agent.policy_store import PolicyStore

    return PolicyStore(DictConfig(config), stats_client=None)


def save_policy_to_store(
    policy_store: Any,
    policy: torch.nn.Module,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Save a policy to the PolicyStore with metadata.

    Args:
        policy_store: PolicyStore instance
        policy: Policy to save
        name: Name for the policy
        metadata: Optional metadata dict

    Returns:
        PolicyRecord instance
    """
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
    """Run a full simulation suite evaluation.

    Args:
        policy_path: Path to policy checkpoint
        suite_name: Name of the simulation suite
        num_envs: Number of environments
        num_episodes: Number of episodes per task
        device: Device to use
        logger: Optional logger

    Returns:
        Dictionary with evaluation results for all tasks
    """
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
    from metta.agent.policy_record import PolicyRecord

    policy_record = PolicyRecord(
        name=os.path.basename(policy_path),
        uri=f"file://{os.path.abspath(policy_path)}",
        generation=1,
    )

    # Run simulation suite
    logger.info(f"Running simulation suite '{suite_name}' with policy: {policy_path}")
    sim_suite = SimulationSuite(
        config=suite_config,
        policy_pr=policy_record,
        device=device,
    )

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

    Args:
        policy_path: Path to policy checkpoint
        num_episodes: Number of episodes to record
        output_dir: Directory to save replays
        device: Device to use
        logger: Optional logger

    Returns:
        List of replay file paths
    """
    if logger is None:
        logger = get_logger("replay_generator")

    # This is a placeholder - actual replay generation would require
    # integration with the replay recording system
    logger.info("Replay generation not yet implemented in functional API")
    return []
