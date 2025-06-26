# mettagrid/src/metta/__init__.py
"""Metta: A modular reinforcement learning library.

This module provides convenient factory functions for creating Metta components
with sensible defaults.
"""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from metta.rl.trainer_config import (
    InitialPolicyConfig,
    KickstartConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    PrioritizedExperienceReplayConfig,
    TrainerConfig,
    VTraceConfig,
)

# Direct instantiation functions (no Hydra required)


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
    """Create a Metta agent instance directly.

    Args:
        obs_space: Observation space
        action_space: Action space
        obs_width: Width of observation
        obs_height: Height of observation
        feature_normalizations: Feature normalization values
        global_features: List of global features
        device: Torch device
        obs_key: Key for observations
        clip_range: Clipping range
        analyze_weights_interval: Interval for weight analysis
        l2_init_weight_update_interval: Interval for L2 init weight updates

    Returns:
        MettaAgent instance
    """
    from metta.agent.metta_agent import MettaAgent

    config = create_agent(obs_key, clip_range, analyze_weights_interval, l2_init_weight_update_interval)

    return MettaAgent(
        obs_space=obs_space,
        obs_width=obs_width,
        obs_height=obs_height,
        action_space=action_space,
        feature_normalizations=feature_normalizations,
        global_features=global_features,
        device=device,
        **config,  # Unpack the config dict as keyword arguments
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
    """Create an optimizer directly.

    Args:
        parameters: Model parameters to optimize
        learning_rate: Learning rate
        beta1: Beta1 for Adam
        beta2: Beta2 for Adam
        eps: Epsilon for Adam
        weight_decay: Weight decay
        type: Optimizer type (currently only "adam" supported)

    Returns:
        Torch optimizer instance
    """
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
    """Create an experience buffer directly.

    Returns:
        Experience instance
    """
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
    """Create a PPO loss module directly.

    Returns:
        ClipPPOLoss instance
    """
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
    """Create a vectorized environment directly.

    Args:
        env_config: Environment configuration dictionary
        num_envs: Number of environments
        num_workers: Number of workers
        batch_size: Batch size for vectorization
        device: Device to use
        zero_copy: Whether to use zero-copy optimization
        vectorization: Vectorization backend (serial/multiprocessing/ray)

    Returns:
        Vectorized environment
    """
    from metta.mettagrid.curriculum.core import SingleTaskCurriculum
    from metta.rl.vecenv import make_vecenv

    # Create a simple curriculum with the env config
    curriculum = SingleTaskCurriculum("task", DictConfig(env_config))

    return make_vecenv(
        curriculum=curriculum,
        vectorization=vectorization,
        num_envs=num_envs,
        batch_size=batch_size,
        num_workers=num_workers,
        zero_copy=zero_copy,
        is_training=True,
    )


# Configuration creation functions (for backward compatibility and Hydra usage)


def create_agent(
    obs_key: str = "grid_obs",
    clip_range: float = 0,
    analyze_weights_interval: int = 300,
    l2_init_weight_update_interval: int = 0,
) -> Dict[str, Any]:
    """Create a default Metta agent configuration.

    Returns a configuration dict that can be used with hydra.utils.instantiate
    to create a MettaAgent instance.
    """
    return {
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


def create_env(
    num_agents: int = 2,
    width: int = 15,
    height: int = 10,
    max_steps: int = 1000,
    obs_width: int = 11,
    obs_height: int = 11,
) -> Dict[str, Any]:
    """Create a default MetaGrid environment configuration.

    Returns a configuration dict for a simple resource gathering environment.
    """
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
                    "heart_max": 1000,
                },
            },
            "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
            "objects": {
                "altar": {
                    "input_battery.red": 1,
                    "output_heart": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_items": 1,
                },
                "mine_red": {
                    "output_ore.red": 1,
                    "color": 0,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 50,
                    "initial_items": 1,
                },
                "generator_red": {
                    "input_ore.red": 1,
                    "output_battery.red": 1,
                    "color": 0,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 50,
                    "initial_items": 1,
                },
                "wall": {"swappable": False},
                "block": {"swappable": True},
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


def create_optimizer(
    learning_rate: float = 0.0004573146765703167,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-12,
    weight_decay: float = 0,
) -> OptimizerConfig:
    """Create a default optimizer configuration."""
    return OptimizerConfig(
        type="adam",
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )


def create_lr_scheduler(
    enabled: bool = False,
    anneal_lr: bool = False,
    warmup_steps: Optional[int] = None,
    schedule_type: Optional[str] = None,
) -> LRSchedulerConfig:
    """Create a default learning rate scheduler configuration."""
    return LRSchedulerConfig(
        enabled=enabled,
        anneal_lr=anneal_lr,
        warmup_steps=warmup_steps,
        schedule_type=schedule_type,
    )


def create_prioritized_replay(
    prio_alpha: float = 0.0,
    prio_beta0: float = 0.6,
) -> PrioritizedExperienceReplayConfig:
    """Create a default prioritized experience replay configuration."""
    return PrioritizedExperienceReplayConfig(
        prio_alpha=prio_alpha,
        prio_beta0=prio_beta0,
    )


def create_vtrace(
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
) -> VTraceConfig:
    """Create a default V-trace configuration."""
    return VTraceConfig(
        vtrace_rho_clip=vtrace_rho_clip,
        vtrace_c_clip=vtrace_c_clip,
    )


def create_kickstart(
    teacher_uri: Optional[str] = None,
    action_loss_coef: float = 1,
    value_loss_coef: float = 1,
    anneal_ratio: float = 0.65,
    kickstart_steps: int = 1_000_000_000,
) -> KickstartConfig:
    """Create a default kickstart configuration."""
    return KickstartConfig(
        teacher_uri=teacher_uri,
        action_loss_coef=action_loss_coef,
        value_loss_coef=value_loss_coef,
        anneal_ratio=anneal_ratio,
        kickstart_steps=kickstart_steps,
        additional_teachers=None,
    )


def create_initial_policy(
    uri: Optional[str] = None,
    type: str = "latest",
    range: int = 10,
    metric: str = "epoch",
) -> InitialPolicyConfig:
    """Create a default initial policy configuration."""
    return InitialPolicyConfig(
        uri=uri,
        type=type,
        range=range,
        metric=metric,
        filters={},
    )


def create_trainer(
    total_timesteps: int = 10_000,
    batch_size: int = 256,
    checkpoint_dir: str = "./checkpoints",
    num_workers: int = 1,
    **kwargs,
) -> TrainerConfig:
    """Create a default trainer configuration.

    Args:
        total_timesteps: Total number of timesteps to train
        batch_size: Batch size for training
        checkpoint_dir: Directory to save checkpoints
        num_workers: Number of workers for data collection
        **kwargs: Additional keyword arguments to override defaults

    Returns:
        TrainerConfig instance with sensible defaults
    """
    # Calculate appropriate minibatch_size based on batch_size
    minibatch_size = min(32, batch_size)
    while batch_size % minibatch_size != 0 and minibatch_size > 1:
        minibatch_size -= 1

    defaults = {
        "target": "metta.rl.trainer.MettaTrainer",
        "total_timesteps": total_timesteps,
        "clip_coef": 0.1,
        "ent_coef": 0.0021,
        "gae_lambda": 0.916,
        "gamma": 0.977,
        "optimizer": create_optimizer(),
        "lr_scheduler": create_lr_scheduler(),
        "max_grad_norm": 0.5,
        "vf_clip_coef": 0.1,
        "vf_coef": 0.44,
        "l2_reg_loss_coef": 0,
        "l2_init_loss_coef": 0,
        "prioritized_experience_replay": create_prioritized_replay(),
        "norm_adv": True,
        "clip_vloss": True,
        "target_kl": None,
        "vtrace": create_vtrace(),
        "zero_copy": True,
        "require_contiguous_env_ids": False,
        "verbose": True,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "bptt_horizon": 8,
        "update_epochs": 1,
        "cpu_offload": False,
        "compile": False,
        "compile_mode": "reduce-overhead",
        "profiler_interval_epochs": 10000,
        "forward_pass_minibatch_target_size": 32,
        "async_factor": 1,
        "kickstart": create_kickstart(),
        "env_overrides": {},
        "num_workers": num_workers,
        "env": None,
        "curriculum": "simple_task",
        "initial_policy": create_initial_policy(),
        "checkpoint_dir": checkpoint_dir,
        "evaluate_interval": 10000,
        "checkpoint_interval": 100,
        "wandb_checkpoint_interval": 1000,
        "replay_interval": 10000,
        "replay_dir": "s3://softmax-public/replays/default",
        "grad_mean_variance_interval": 0,
    }

    # Override defaults with any provided kwargs
    defaults.update(kwargs)

    return TrainerConfig(**defaults)


def create_sim_suite(
    name: str = "all",
    num_envs: int = 4,
    num_episodes: int = 2,
    map_preview_limit: int = 32,
) -> Dict[str, Any]:
    """Create a default simulation suite configuration."""
    return {
        "_target_": "metta.sim.simulation_config.SimulationSuiteConfig",
        "name": name,
        "num_envs": num_envs,
        "num_episodes": num_episodes,
        "map_preview_limit": map_preview_limit,
        "suites": [],
    }


def create_wandb(
    mode: str = "disabled",
    project: str = "metta",
    entity: Optional[str] = None,
    tags: Optional[list] = None,
) -> Dict[str, Any]:
    """Create a default WandB configuration."""
    return {
        "mode": mode,
        "project": project,
        "entity": entity,
        "tags": tags or [],
    }
