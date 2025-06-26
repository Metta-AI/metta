#!/usr/bin/env python3
"""
Universal entry point for Metta - Alternative to tools/* scripts without Hydra

This script demonstrates how to use Metta as a library without relying on Hydra
configuration. It replicates the functionality of:
- tools/train.py (default)
- tools/sim.py
- tools/analyze.py
- tools/dashboard.py

Usage:
    # Train with default settings
    python run.py

    # Train with custom settings
    python run.py train --run my_experiment --total-timesteps 1000000

    # Evaluate a policy
    python run.py sim --run my_experiment --policy-uri file://./train_dir/my_experiment/checkpoints

    # Analyze a policy
    python run.py analyze --policy-uri file://./train_dir/my_experiment/checkpoints/policy_v1.pt

    # Generate dashboard
    python run.py dashboard --output-path ./dashboard_data.json
"""

import argparse
import json
import os
import sys

import torch
from omegaconf import DictConfig, OmegaConf

# Import basic requirements first
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.rl.experience import Experience
from metta.rl.functional_trainer import rollout, train_ppo
from metta.rl.losses import Losses
from metta.rl.trainer_config import (
    InitialPolicyConfig,
    KickstartConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    PrioritizedExperienceReplayConfig,
    TrainerConfig,
    VTraceConfig,
)
from metta.rl.vecenv import make_vecenv


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


def train_command(args):
    """Execute training using functional trainer API."""
    import torch
    from hydra.utils import instantiate

    from metta.agent.policy_store import PolicyStore
    from metta.common.stopwatch import Stopwatch

    # Import pufferlib C extensions for torch.ops.pufferlib
    try:
        from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib
    except ImportError:
        raise ImportError(
            "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
            "try installing with --no-build-isolation"
        ) from None

    cfg = build_train_config(args)
    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("train")
    logger.info(f"Training configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)[:500]}...")
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(f"{cfg.run_dir}/checkpoints", exist_ok=True)
    policy_store = PolicyStore(cfg, None)
    curriculum = SingleTaskCurriculum("simple_task", cfg.env)

    try:
        trainer_cfg = cfg.trainer
        device = torch.device(cfg.device)
        total_timesteps = trainer_cfg.total_timesteps
        logger.info(
            f"Original trainer config: batch_size={trainer_cfg.batch_size}, minibatch_size={trainer_cfg.minibatch_size}, bptt_horizon={trainer_cfg.bptt_horizon}"
        )
        logger.info("Creating vectorized environment...")
        num_agents = curriculum.get_task().env_cfg().game.num_agents
        target_batch_size = trainer_cfg.forward_pass_minibatch_target_size // num_agents
        if target_batch_size < max(2, trainer_cfg.num_workers):
            target_batch_size = trainer_cfg.num_workers
        forward_batch_size = (target_batch_size // trainer_cfg.num_workers) * trainer_cfg.num_workers
        num_envs = forward_batch_size * trainer_cfg.async_factor
        logger.info(
            f"Creating {num_envs} environments (forward_batch_size={forward_batch_size}, "
            f"async_factor={trainer_cfg.async_factor}, num_workers={trainer_cfg.num_workers})"
        )
        vecenv_batch_size = None if trainer_cfg.num_workers == 1 else forward_batch_size
        vecenv = make_vecenv(
            curriculum=curriculum,
            vectorization=cfg.vectorization,
            num_envs=num_envs,
            batch_size=vecenv_batch_size,
            num_workers=trainer_cfg.num_workers,
            zero_copy=trainer_cfg.zero_copy,
            is_training=True,
        )
        metta_grid_env = vecenv.driver_env
        actions_names = metta_grid_env.action_names
        actions_max_params = metta_grid_env.max_action_args
        logger.info("Creating agent...")
        import gymnasium as gym
        import numpy as np

        obs_space = gym.spaces.Dict(
            {
                "grid_obs": metta_grid_env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )
        agent_config = dict(cfg.agent)
        policy = instantiate(
            agent_config,
            obs_space=obs_space,
            obs_width=metta_grid_env.obs_width,
            obs_height=metta_grid_env.obs_height,
            action_space=metta_grid_env.single_action_space,
            feature_normalizations=metta_grid_env.feature_normalizations,
            global_features=metta_grid_env.global_features,
            device=device,
            _recursive_=False,
            _convert_="all",
        )
        policy.activate_actions(actions_names, actions_max_params, device)
        logger.info("Creating experience buffer...")
        bptt_horizon = trainer_cfg.get("bptt_horizon", 8)
        total_agents = vecenv.num_agents
        world_size = 1
        batch_size = trainer_cfg.batch_size // world_size
        minibatch_size = trainer_cfg.minibatch_size // world_size
        logger.info(
            f"Creating experience buffer with batch_size={batch_size}, minibatch_size={minibatch_size}, total_agents={total_agents}"
        )
        experience = Experience(
            total_agents=total_agents,
            batch_size=batch_size,
            bptt_horizon=bptt_horizon,
            minibatch_size=minibatch_size,
            max_minibatch_size=trainer_cfg.get("forward_pass_minibatch_target_size", 2048),
            obs_space=metta_grid_env.single_observation_space,
            atn_space=metta_grid_env.single_action_space,
            device=device,
            hidden_size=policy.hidden_size,
            cpu_offload=trainer_cfg.cpu_offload,
            num_lstm_layers=policy.core_num_layers,
            agents_per_batch=getattr(vecenv, "agents_per_batch", None),
        )
        timer = Stopwatch(logger)
        timer.start()
        logger.info("Resetting environments...")
        seed = cfg.get("seed", 42)
        vecenv.async_reset(seed)
        logger.info("Starting training loop...")
        agent_step = 0
        epoch = 0
        while agent_step < total_timesteps:
            steps_before = agent_step
            with timer("rollout"):
                agent_step, rollout_stats = rollout(
                    policy=policy,
                    vecenv=vecenv,
                    experience=experience,
                    device=device,
                    agent_step=agent_step,
                    timer=timer,
                )
            with timer("train"):
                if epoch == 0:
                    optimizer = torch.optim.Adam(
                        policy.parameters(),
                        lr=trainer_cfg.optimizer.learning_rate,
                        betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
                        eps=trainer_cfg.optimizer.eps,
                        weight_decay=trainer_cfg.optimizer.weight_decay,
                    )
                    losses = Losses()
                epoch = train_ppo(
                    policy=policy,
                    optimizer=optimizer,
                    experience=experience,
                    device=device,
                    losses=losses,
                    epoch=epoch,
                    cfg=cfg,
                    gamma=trainer_cfg.gamma,
                    gae_lambda=trainer_cfg.gae_lambda,
                    clip_coef=trainer_cfg.clip_coef,
                    ent_coef=trainer_cfg.ent_coef,
                    vf_coef=trainer_cfg.vf_coef,
                    max_grad_norm=trainer_cfg.max_grad_norm,
                    norm_adv=trainer_cfg.norm_adv,
                    clip_vloss=trainer_cfg.clip_vloss,
                    vf_clip_coef=trainer_cfg.vf_clip_coef,
                    update_epochs=trainer_cfg.update_epochs,
                    agent_step=agent_step,
                    total_timesteps=total_timesteps,
                    batch_size=batch_size,
                    vtrace_rho_clip=trainer_cfg.vtrace.vtrace_rho_clip,
                    vtrace_c_clip=trainer_cfg.vtrace.vtrace_c_clip,
                )
            steps_in_epoch = agent_step - steps_before
            rollout_time = timer.get_last_elapsed("rollout")
            train_time = timer.get_last_elapsed("train")
            total_time = rollout_time + train_time
            steps_per_sec = steps_in_epoch / total_time if total_time > 0 else 0
            loss_stats = losses.stats()
            logger.info(
                f"Epoch {epoch} - Agent steps: {agent_step}/{total_timesteps} - "
                f"{steps_per_sec:.0f} steps/sec - "
                f"Policy loss: {loss_stats['policy_loss']:.4f} - "
                f"Value loss: {loss_stats['value_loss']:.4f}"
            )
            if epoch % trainer_cfg.checkpoint_interval == 0:
                checkpoint_path = f"{cfg.run_dir}/checkpoints/policy_epoch_{epoch}.pt"
                torch.save(policy.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            epoch += 1
        final_checkpoint = f"{cfg.run_dir}/checkpoints/policy_final.pt"
        torch.save(policy.state_dict(), final_checkpoint)
        logger.info(f"Training complete! Final checkpoint saved to {final_checkpoint}")
        vecenv.close()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


def sim_command(args):
    """Execute simulation/evaluation."""
    # Import simulation dependencies only when needed
    from metta.agent.policy_store import PolicyStore
    from metta.sim.simulation_config import SimulationSuiteConfig
    from metta.sim.simulation_suite import SimulationSuite

    cfg = build_common_config(args)

    # Build sim job configuration
    sim_job_config = {
        "policy_uris": [args.policy_uri],
        "simulation_suite": {
            "name": "eval",
            "num_envs": getattr(args, "num_envs", 32),
            "num_episodes": getattr(args, "num_episodes", 10),
            "map_preview_limit": 32,
            "suites": [],
        },
        "stats_dir": f"{cfg.run_dir}/stats",
        "stats_db_uri": f"{cfg.run_dir}/stats.db",
        "replay_dir": f"{cfg.run_dir}/replays/evals",
        "selector_type": getattr(args, "selector_type", "top"),
    }

    cfg["sim_job"] = DictConfig(sim_job_config)

    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("sim")

    logger.info(f"Simulation configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

    # Create output directories
    os.makedirs(sim_job_config["stats_dir"], exist_ok=True)
    os.makedirs(sim_job_config["replay_dir"], exist_ok=True)

    policy_store = PolicyStore(cfg, None)

    # Load and evaluate policies
    results = {"policies": []}

    for policy_uri in sim_job_config["policy_uris"]:
        metric = sim_job_config["simulation_suite"]["name"] + "_score"
        policy_prs = policy_store.policies(policy_uri, sim_job_config["selector_type"], n=1, metric=metric)

        for pr in policy_prs:
            logger.info(f"Evaluating policy {pr.uri}")

            replay_dir = f"{sim_job_config['replay_dir']}/{pr.name}"
            sim = SimulationSuite(
                config=SimulationSuiteConfig(sim_job_config["simulation_suite"]),
                policy_pr=pr,
                policy_store=policy_store,
                replay_dir=replay_dir,
                stats_dir=sim_job_config["stats_dir"],
                device=cfg.device,
                vectorization=cfg.vectorization,
                stats_client=None,
            )

            sim_results = sim.simulate()

            # Collect results
            checkpoint_data = {"name": pr.name, "uri": pr.uri, "metrics": {}}

            # Get average reward
            rewards_df = sim_results.stats_db.query(
                "SELECT AVG(value) AS reward_avg FROM agent_metrics WHERE metric = 'reward'"
            )
            if len(rewards_df) > 0 and rewards_df.iloc[0]["reward_avg"] is not None:
                checkpoint_data["metrics"]["reward_avg"] = float(rewards_df.iloc[0]["reward_avg"])

            results["policies"].append(checkpoint_data)

            # Export stats DB
            logger.info(f"Exporting stats DB to {sim_job_config['stats_db_uri']}")
            sim_results.stats_db.export(sim_job_config["stats_db_uri"])

    # Output results
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


def analyze_command(args):
    """Execute analysis."""
    try:
        # Import analysis dependencies only when needed
        from metta.agent.policy_store import PolicyStore
        from metta.eval.analysis import analyze
        from metta.eval.analysis_config import AnalysisConfig
    except Exception as e:
        print(f"Error importing analysis modules: {e}")
        print("Analysis functionality is currently unavailable.")
        print("Please use tools/analyze.py with Hydra configuration.")
        sys.exit(1)

    cfg = build_common_config(args)

    # Build analysis configuration
    analysis_config = {
        "policy_uri": args.policy_uri,
        "policy_selector": {
            "type": getattr(args, "selector_type", "top"),
            "metric": getattr(args, "metric", "reward_avg"),
        },
        "output_dir": getattr(args, "output_dir", f"{cfg.run_dir}/analysis"),
        "num_episodes": getattr(args, "num_episodes", 10),
        "num_envs": getattr(args, "num_envs", 1),
    }

    cfg["analysis"] = DictConfig(analysis_config)

    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("analyze")

    logger.info(f"Analysis configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

    # Create output directory
    os.makedirs(analysis_config["output_dir"], exist_ok=True)

    policy_store = PolicyStore(cfg, None)

    # Load policy
    policy_pr = policy_store.policy(
        analysis_config["policy_uri"],
        analysis_config["policy_selector"]["type"],
        metric=analysis_config["policy_selector"]["metric"],
    )

    # Run analysis
    analyze(policy_pr, AnalysisConfig(cfg.analysis))

    logger.info(f"Analysis complete. Results saved to {analysis_config['output_dir']}")


def dashboard_command(args):
    """Execute dashboard generation."""
    try:
        # Import dashboard dependencies only when needed
        from metta.eval.dashboard_data import DashboardConfig, write_dashboard_data
    except Exception as e:
        print(f"Error importing dashboard modules: {e}")
        print("Dashboard functionality is currently unavailable.")
        print("Please use tools/dashboard.py with Hydra configuration.")
        sys.exit(1)

    cfg = build_common_config(args)

    # Build dashboard configuration
    dashboard_config = {
        "stats_db_uris": getattr(args, "stats_db_uris", []),
        "output_path": args.output_path,
        "include_replays": getattr(args, "include_replays", False),
    }

    cfg["dashboard"] = DictConfig(dashboard_config)

    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("dashboard")

    logger.info(f"Dashboard configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

    # Generate dashboard data
    write_dashboard_data(DashboardConfig(cfg.dashboard))

    logger.info(f"Dashboard data written to {dashboard_config['output_path']}")

    if dashboard_config["output_path"].startswith("s3://"):
        from metta.mettagrid.util.file import http_url

        dashboard_url = "https://metta-ai.github.io/metta/observatory/"
        logger.info(f"View dashboard at {dashboard_url}?data={http_url(dashboard_config['output_path'])}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Universal entry point for Metta - Alternative to tools/* scripts without Hydra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (default command)
  python run.py

  # Train with custom settings
  python run.py train --run my_experiment --total-timesteps 1000000

  # Evaluate a policy
  python run.py sim --run my_experiment --policy-uri file://./train_dir/my_experiment/checkpoints

  # Analyze a policy
  python run.py analyze --policy-uri file://./train_dir/my_experiment/checkpoints/policy_v1.pt

  # Generate dashboard
  python run.py dashboard --output-path ./dashboard_data.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument("--run", default="default_run", help="Experiment name")
    train_parser.add_argument("--total-timesteps", type=int, default=10_000, help="Total training timesteps")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    train_parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    train_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    train_parser.add_argument(
        "--vectorization", default="serial", help="Vectorization mode (serial/multiprocessing/ray)"
    )
    train_parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")

    # Sim command
    sim_parser = subparsers.add_parser("sim", help="Simulate/evaluate a policy")
    sim_parser.add_argument("--run", required=True, help="Experiment name")
    sim_parser.add_argument("--policy-uri", required=True, help="Policy URI to evaluate")
    sim_parser.add_argument("--num-envs", type=int, default=32, help="Number of environments")
    sim_parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    sim_parser.add_argument("--selector-type", default="top", help="Policy selector type")
    sim_parser.add_argument("--device", default="cuda", help="Device to use")
    sim_parser.add_argument("--vectorization", default="multiprocessing", help="Vectorization backend")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a policy")
    analyze_parser.add_argument("--policy-uri", required=True, help="Policy URI to analyze")
    analyze_parser.add_argument("--output-dir", help="Output directory for analysis")
    analyze_parser.add_argument("--selector-type", default="top", help="Policy selector type")
    analyze_parser.add_argument("--metric", default="reward_avg", help="Metric for policy selection")
    analyze_parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    analyze_parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    analyze_parser.add_argument("--device", default="cuda", help="Device to use")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Generate dashboard data")
    dashboard_parser.add_argument("--output-path", required=True, help="Output path for dashboard data")
    dashboard_parser.add_argument("--stats-db-uris", nargs="+", default=[], help="Stats DB URIs to include")
    dashboard_parser.add_argument("--include-replays", action="store_true", help="Include replay data")

    args = parser.parse_args()

    # Default to train command if no command specified
    if not args.command:
        args.command = "train"
        # Create a namespace with default train arguments
        args.run = "default_run"
        args.total_timesteps = 10_000
        args.batch_size = 256
        args.num_agents = 2
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.seed = 0
        args.vectorization = "serial"
        args.num_workers = 1

    # Execute the appropriate command
    if args.command == "train":
        train_command(args)
    elif args.command == "sim":
        sim_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    elif args.command == "dashboard":
        dashboard_command(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
