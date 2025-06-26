import os

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


def build_common_config(args):
    """Build the common configuration that all tools share."""
    data_dir = os.environ.get("DATA_DIR", "./train_dir")

    cfg = {
        "run": getattr(args, "run", "default_run"),
        "data_dir": data_dir,
        "run_dir": f"{data_dir}/{getattr(args, 'run', 'default_run')}",
        "policy_uri": f"file://{data_dir}/{getattr(args, 'run', 'default_run')}/checkpoints",
        "torch_deterministic": True,
        "vectorization": getattr(args, "vectorization", "multiprocessing"),
        "seed": getattr(args, "seed", 0),
        "device": getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"),
        "stats_user": os.environ.get("USER", "unknown"),
        "dist_cfg_path": None,
        # Hydra config for compatibility
        "hydra": {"callbacks": {"resolver_callback": {"_target_": "metta.common.util.resolvers.ResolverRegistrar"}}},
    }

    return DictConfig(cfg)


def build_train_config(args):
    """Build configuration for training."""
    cfg = build_common_config(args)

    # Environment configuration - using Python format that will be converted by MettaGridEnv
    env_config = {
        "sampling": 0,
        "desync_episodes": False,
        "replay_level_prob": 0.0,  # Set to 0 for simpler initial testing
        "game": {
            "num_agents": getattr(args, "num_agents", 2),  # Start with just 2 agents
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            "max_steps": 1000,
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
            # Groups in Python format - MettaGridEnv will convert to agent_groups
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
                "attack": {"enabled": False},  # Disabled for simpler testing
                "swap": {"enabled": True},
                "change_color": {"enabled": False},
            },
            "reward_sharing": {
                "groups": {}  # Empty for single agent group
            },
            "map_builder": {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": 15,
                "height": 10,
                "border_width": 2,
                "agents": getattr(args, "num_agents", 2),
                "objects": {"mine_red": 2, "generator_red": 1, "altar": 1, "wall": 5, "block": 3},
            },
        },
    }

    # Agent configuration (based on configs/agent/fast.yaml)
    agent_config = {
        "_target_": "metta.agent.metta_agent.MettaAgent",
        "observations": {"obs_key": "grid_obs"},
        "clip_range": 0,
        "analyze_weights_interval": 300,
        "l2_init_weight_update_interval": 0,
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

    # Add env configuration to main config first
    cfg["env"] = DictConfig(env_config)

    # Trainer configuration using typed configs
    optimizer_config = OptimizerConfig(
        type="adam",
        learning_rate=0.0004573146765703167,
        beta1=0.9,
        beta2=0.999,
        eps=1e-12,
        weight_decay=0,
    )

    lr_scheduler_config = LRSchedulerConfig(
        enabled=False,
        anneal_lr=False,
        warmup_steps=None,
        schedule_type=None,
    )

    prioritized_replay_config = PrioritizedExperienceReplayConfig(
        prio_alpha=0.0,
        prio_beta0=0.6,
    )

    vtrace_config = VTraceConfig(
        vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0,
    )

    kickstart_config = KickstartConfig(
        teacher_uri=None,
        action_loss_coef=1,
        value_loss_coef=1,
        anneal_ratio=0.65,
        kickstart_steps=1_000_000_000,
        additional_teachers=None,
    )

    initial_policy_config = InitialPolicyConfig(
        uri=None,
        type="latest",
        range=10,
        metric="epoch",
        filters={},
    )

    # Calculate appropriate minibatch_size based on batch_size
    batch_size = getattr(args, "batch_size", 256)
    minibatch_size = min(32, batch_size)  # Default to 32 but cap at batch_size
    # Ensure batch_size is divisible by minibatch_size
    while batch_size % minibatch_size != 0 and minibatch_size > 1:
        minibatch_size -= 1

    # Create the main trainer config
    trainer_config = TrainerConfig(
        target="metta.rl.trainer.MettaTrainer",
        total_timesteps=getattr(args, "total_timesteps", 10_000),
        clip_coef=0.1,
        ent_coef=0.0021,
        gae_lambda=0.916,
        gamma=0.977,
        optimizer=optimizer_config,
        lr_scheduler=lr_scheduler_config,
        max_grad_norm=0.5,
        vf_clip_coef=0.1,
        vf_coef=0.44,
        l2_reg_loss_coef=0,
        l2_init_loss_coef=0,
        prioritized_experience_replay=prioritized_replay_config,
        norm_adv=True,
        clip_vloss=True,
        target_kl=None,
        vtrace=vtrace_config,
        zero_copy=True,
        require_contiguous_env_ids=False,
        verbose=True,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        bptt_horizon=8,
        update_epochs=1,
        cpu_offload=False,
        compile=False,
        compile_mode="reduce-overhead",
        profiler_interval_epochs=10000,
        forward_pass_minibatch_target_size=32,
        async_factor=1,
        kickstart=kickstart_config,
        env_overrides={},
        num_workers=getattr(args, "num_workers", 1),
        env=None,  # Will be set by curriculum
        curriculum="simple_task",  # Dummy curriculum name
        initial_policy=initial_policy_config,
        checkpoint_dir=f"{cfg.run_dir}/checkpoints",
        evaluate_interval=10000,  # Set to high value to effectively disable
        checkpoint_interval=100,
        wandb_checkpoint_interval=1000,
        replay_interval=10000,  # Set to high value to effectively disable
        replay_dir=f"s3://softmax-public/replays/{cfg.run}",
        grad_mean_variance_interval=0,
    )

    # Convert to dict and add extra fields not in TrainerConfig
    trainer_dict = trainer_config.model_dump(by_alias=True)
    trainer_dict["resume"] = False
    trainer_dict["use_e3b"] = False
    trainer_dict["replay_uri"] = None

    # Simulation suite configuration for evals
    sim_config = {
        "_target_": "metta.sim.simulation_config.SimulationSuiteConfig",
        "name": "all",
        "num_envs": 4,  # Small for testing
        "num_episodes": 2,
        "map_preview_limit": 32,
        "suites": [],
    }

    # Add configurations to main config
    cfg["agent"] = DictConfig(agent_config)
    cfg["trainer"] = DictConfig(trainer_dict)
    cfg["sim"] = DictConfig(sim_config)

    # WandB configuration
    cfg["wandb"] = DictConfig(
        {
            "mode": "disabled",  # Can be overridden with --wandb-mode
            "project": "metta",
            "entity": None,
            "tags": [],
        }
    )

    # Train job configuration
    cfg["train_job"] = DictConfig({"map_preview_uri": None, "evals": cfg.sim})

    cfg["cmd"] = "train"

    # Set serial vectorization for local
    cfg["vectorization"] = getattr(args, "vectorization", "serial")

    return cfg
