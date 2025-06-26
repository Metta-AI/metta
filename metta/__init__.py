# metta/__init__.py
"""Metta: A modular reinforcement learning library.

This module provides convenient factory functions for creating Metta components
with sensible defaults.
"""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from typing import Any, Dict, Optional

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


def agent(
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


def env(
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


def optimizer(
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


def lr_scheduler(
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


def prioritized_replay(
    prio_alpha: float = 0.0,
    prio_beta0: float = 0.6,
) -> PrioritizedExperienceReplayConfig:
    """Create a default prioritized experience replay configuration."""
    return PrioritizedExperienceReplayConfig(
        prio_alpha=prio_alpha,
        prio_beta0=prio_beta0,
    )


def vtrace(
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
) -> VTraceConfig:
    """Create a default V-trace configuration."""
    return VTraceConfig(
        vtrace_rho_clip=vtrace_rho_clip,
        vtrace_c_clip=vtrace_c_clip,
    )


def kickstart(
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


def initial_policy(
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


def trainer(
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
        "optimizer": optimizer(),
        "lr_scheduler": lr_scheduler(),
        "max_grad_norm": 0.5,
        "vf_clip_coef": 0.1,
        "vf_coef": 0.44,
        "l2_reg_loss_coef": 0,
        "l2_init_loss_coef": 0,
        "prioritized_experience_replay": prioritized_replay(),
        "norm_adv": True,
        "clip_vloss": True,
        "target_kl": None,
        "vtrace": vtrace(),
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
        "kickstart": kickstart(),
        "env_overrides": {},
        "num_workers": num_workers,
        "env": None,
        "curriculum": "simple_task",
        "initial_policy": initial_policy(),
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


def sim_suite(
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


def wandb(
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
