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
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment


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
        "hydra": {"callbacks": {"resolver_callback": {"_target_": "metta.util.resolvers.ResolverRegistrar"}}},
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

    # Agent configuration
    agent_config = {
        "_target_": "metta.agent.metta_agent.MettaAgent",
        "hidden_size": 128,  # Reduced for faster testing
        "rl_layer": {"_target_": "metta.agent.lib.rl_layer.SimpleLSTMRLLayer", "hidden_size": 128, "num_layers": 1},
        "torso": {
            "_target_": "metta.agent.lib.torso.SimpleTorso",
            "hidden_size": 128,
            "resnet_channels": 32,
            "num_resnet_blocks": 2,
            "num_heads": 0,
        },
        "heads": {
            "actor": {
                "_target_": "metta.agent.lib.head.LinearHead",
                "input_size": 128,
                "output_size": "???",
                "num_layers": 0,
            },
            "critic": {
                "_target_": "metta.agent.lib.head.LinearHead",
                "input_size": 128,
                "output_size": 1,
                "num_layers": 0,
            },
        },
        "decoder": {"_target_": "metta.agent.lib.decoder.MultiDiscreteActionDecoder"},
        "sample_dtype": "float32",
        "device": cfg.device,
        "normalize_observations": False,
    }

    # Add env configuration to main config first
    cfg["env"] = DictConfig(env_config)

    # Trainer configuration (based on configs/trainer/puffer.yaml)
    trainer_config = {
        "_target_": "metta.rl.trainer.MettaTrainer",
        "resume": False,  # Start fresh for testing
        "use_e3b": False,
        "total_timesteps": getattr(args, "total_timesteps", 10_000),  # Small default for testing
        "clip_coef": 0.1,
        "ent_coef": 0.01,  # Increased for exploration
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "optimizer": {
            "type": "adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "learning_rate": 3e-4,
            "weight_decay": 0,
        },
        "lr_scheduler": {"enabled": False, "anneal_lr": False},
        "max_grad_norm": 0.5,
        "vf_clip_coef": None,
        "vf_coef": 0.5,
        "l2_reg_loss_coef": 0,
        "l2_init_loss_coef": 0,
        "prioritized_experience_replay": {"prio_alpha": 0.0, "prio_beta0": 0.6},
        "norm_adv": True,
        "clip_vloss": True,
        "target_kl": None,
        "vtrace": {"vtrace_rho_clip": 1.0, "vtrace_c_clip": 1.0},
        "zero_copy": True,
        "require_contiguous_env_ids": False,
        "verbose": True,
        "batch_size": getattr(args, "batch_size", 32),  # Smaller, aligned batch
        "minibatch_size": 16,  # Half of batch
        "bptt_horizon": 8,  # Smaller horizon
        "update_epochs": 1,
        "cpu_offload": False,
        "compile": False,
        "compile_mode": "reduce-overhead",
        "profiler_interval_epochs": 10000,
        "forward_pass_minibatch_target_size": 32,  # Same as batch_size
        "async_factor": 1,  # Single async factor for simplicity
        "kickstart": {
            "teacher_uri": None,
            "action_loss_coef": 1,
            "value_loss_coef": 1,
            "anneal_ratio": 0.65,
            "kickstart_steps": 1_000_000_000,
            "additional_teachers": [],
        },
        # Required fields
        "env_overrides": {},  # No overrides needed
        "num_workers": getattr(args, "num_workers", 1),  # Single worker for simplicity
        "checkpoint_dir": f"{cfg.run_dir}/checkpoints",
        "checkpoint_interval": 100,
        "evaluate_interval": 0,  # Disable evaluation for now
        "replay_interval": 0,  # Disable replay generation
        "wandb_checkpoint_interval": 1000,
        "replay_uri": None,
    }

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
    cfg["trainer"] = DictConfig(trainer_config)
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

    return cfg


def train_command(args):
    """Execute training using functional trainer API."""
    # Import training dependencies only when needed
    import torch
    from hydra.utils import instantiate

    from metta.agent.policy_store import PolicyStore
    from metta.common.stopwatch import Stopwatch

    # Add debugging wrapper for cpp_config_dict
    # Add debugging wrapper for MettaGridEnv
    from metta.mettagrid import mettagrid_c_config, mettagrid_env
    from metta.mettagrid.curriculum.core import SingleTaskCurriculum
    from metta.rl.experience import Experience
    from metta.rl.functional_trainer import rollout, train_ppo
    from metta.rl.vecenv import make_vecenv

    original_init = mettagrid_env.MettaGridEnv.__init__

    def debug_init(
        self, curriculum, render_mode, level=None, buf=None, stats_writer=None, replay_writer=None, **kwargs
    ):
        logger.info("DEBUG: MettaGridEnv.__init__ called")
        logger.info(f"  curriculum type: {type(curriculum)}")
        logger.info(f"  curriculum: {curriculum}")
        try:
            task = curriculum.get_task()
            logger.info(f"  task type: {type(task)}")
            env_cfg = task.env_cfg()
            logger.info(f"  env_cfg type: {type(env_cfg)}")
            logger.info(f"  env_cfg keys: {list(env_cfg.keys()) if hasattr(env_cfg, 'keys') else 'N/A'}")
            if hasattr(env_cfg, "game"):
                logger.info("  env_cfg.game exists: True")
                logger.info(f"  env_cfg.game type: {type(env_cfg.game)}")
                # Don't convert yet, just check structure
                if hasattr(env_cfg.game, "groups"):
                    logger.info("  env_cfg.game has 'groups' attribute")
        except Exception as e:
            logger.error(f"  Error accessing curriculum/task: {e}")

        return original_init(self, curriculum, render_mode, level, buf, stats_writer, replay_writer, **kwargs)

    mettagrid_env.MettaGridEnv.__init__ = debug_init

    original_cpp_config_dict = mettagrid_c_config.cpp_config_dict

    def debug_cpp_config_dict(game_config_dict):
        logger.info("DEBUG: cpp_config_dict called")
        logger.info(f"  Input type: {type(game_config_dict)}")
        logger.info(f"  Has 'groups'? {'groups' in game_config_dict}")
        try:
            result = original_cpp_config_dict(game_config_dict)
            logger.info("  ✓ cpp_config_dict succeeded")
            return result
        except KeyError as e:
            logger.error(f"  ✗ cpp_config_dict failed with KeyError: {e}")
            logger.error(
                f"  Input keys: {list(game_config_dict.keys()) if hasattr(game_config_dict, 'keys') else 'N/A'}"
            )
            raise

    mettagrid_c_config.cpp_config_dict = debug_cpp_config_dict

    cfg = build_train_config(args)

    # Force serial vectorization for debugging
    cfg.vectorization = "serial"

    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("train")

    logger.info(f"Training configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)[:500]}...")

    # Create output directories
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(f"{cfg.run_dir}/checkpoints", exist_ok=True)

    # Initialize policy store
    policy_store = PolicyStore(cfg, None)

    # Create curriculum directly
    curriculum = SingleTaskCurriculum("simple_task", cfg.env)

    try:
        # Training parameters from config
        trainer_cfg = cfg.trainer
        device = torch.device(cfg.device)
        batch_size = trainer_cfg.batch_size
        minibatch_size = trainer_cfg.minibatch_size
        total_timesteps = trainer_cfg.total_timesteps

        # Create vectorized environment
        logger.info("Creating vectorized environment...")
        vecenv = make_vecenv(
            curriculum=curriculum,
            vectorization=cfg.vectorization,
            num_envs=trainer_cfg.batch_size // trainer_cfg.num_workers,
            num_workers=trainer_cfg.num_workers,
            batch_size=trainer_cfg.batch_size,
            render_mode=None,
        )

        # Get environment info from driver env
        metta_grid_env = vecenv.driver_env
        actions_names = metta_grid_env.action_names
        actions_max_params = metta_grid_env.max_action_args

        # Create agent/policy - instantiate from config
        logger.info("Creating agent...")
        agent_config = dict(cfg.agent)
        agent_config["device"] = device
        policy = instantiate(agent_config, _convert_="all")

        # Activate actions on the policy
        policy.activate_actions(actions_names, actions_max_params, device)

        # Create experience buffer
        logger.info("Creating experience buffer...")
        experience = Experience(
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            device=device,
            lstm=policy.lstm if hasattr(policy, "lstm") else None,
        )

        # Create timer
        timer = Stopwatch(logger)
        timer.start()

        # Training loop
        logger.info("Starting training loop...")
        agent_step = 0
        epoch = 0

        while agent_step < total_timesteps:
            steps_before = agent_step

            # ROLLOUT: Collect experience
            with timer("rollout"):
                agent_step, rollout_stats = rollout(
                    policy=policy,
                    vecenv=vecenv,
                    experience=experience,
                    device=device,
                    agent_step=agent_step,
                    timer=timer,
                )

            # TRAIN: Update policy using PPO
            with timer("train"):
                losses = train_ppo(
                    policy=policy,
                    experience=experience,
                    device=device,
                    learning_rate=trainer_cfg.optimizer.learning_rate,
                    clip_coef=trainer_cfg.clip_coef,
                    value_coef=trainer_cfg.vf_coef,
                    entropy_coef=trainer_cfg.ent_coef,
                    max_grad_norm=trainer_cfg.max_grad_norm,
                    batch_size=batch_size,
                    minibatch_size=minibatch_size,
                    update_epochs=trainer_cfg.update_epochs,
                    norm_adv=trainer_cfg.norm_adv,
                    clip_vloss=trainer_cfg.clip_vloss,
                    timer=timer,
                )

            # Log progress
            steps_in_epoch = agent_step - steps_before
            rollout_time = timer.get_last_elapsed("rollout")
            train_time = timer.get_last_elapsed("train")
            total_time = rollout_time + train_time
            steps_per_sec = steps_in_epoch / total_time if total_time > 0 else 0

            logger.info(
                f"Epoch {epoch} - Agent steps: {agent_step}/{total_timesteps} - "
                f"{steps_per_sec:.0f} steps/sec - "
                f"Loss: {losses.loss:.4f}"
            )

            # Save checkpoint periodically
            if epoch % trainer_cfg.checkpoint_interval == 0:
                checkpoint_path = f"{cfg.run_dir}/checkpoints/policy_epoch_{epoch}.pt"
                torch.save(policy.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            epoch += 1

        # Save final checkpoint
        final_checkpoint = f"{cfg.run_dir}/checkpoints/policy_final.pt"
        torch.save(policy.state_dict(), final_checkpoint)
        logger.info(f"Training complete! Final checkpoint saved to {final_checkpoint}")

        # Clean up
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
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    train_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    train_parser.add_argument("--vectorization", default="multiprocessing", help="Vectorization backend")

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
        args.batch_size = 32
        args.num_agents = 2
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.seed = 0
        args.vectorization = "multiprocessing"

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
