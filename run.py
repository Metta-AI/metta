#!/usr/bin/env python3
"""
Universal entry point for Metta - Alternative to tools/* scripts without Hydra

This script demonstrates how to use Metta as a library without relying on Hydra
configuration. It replicates the functionality of:
- tools/train.py
- tools/sim.py
- tools/analyze.py
- tools/dashboard.py

Usage:
    python run.py train --run my_experiment --total-timesteps 1000000
    python run.py sim --run my_experiment --policy-uri file://./checkpoints
    python run.py analyze --policy-uri file://./checkpoints/policy_v1.pt
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
        "run": args.run if hasattr(args, "run") else "default_run",
        "data_dir": data_dir,
        "run_dir": f"{data_dir}/{args.run if hasattr(args, 'run') else 'default_run'}",
        "policy_uri": f"file://{data_dir}/{args.run if hasattr(args, 'run') else 'default_run'}/checkpoints",
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

    # Environment configuration - simplified version of configs/env/mettagrid/simple.yaml
    env_config = {
        "sampling": 0,
        "desync_episodes": False,
        "replay_level_prob": 0.2,
        "game": {
            "num_agents": args.num_agents if hasattr(args, "num_agents") else 24,
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
                    "ore.red": 0.005,
                    "ore.blue": 0.005,
                    "ore.green": 0.005,
                    "battery.red": 0.01,
                    "battery.blue": 0.01,
                    "battery.green": 0.01,
                    "heart": 1,
                    "heart_max": 1000,
                },
            },
            "groups": {
                "agent": {"id": 0, "sprite": 0, "props": {}},
                "team_1": {"id": 1, "sprite": 1, "group_reward_pct": 0.5, "props": {}},
                "team_2": {"id": 2, "sprite": 4, "group_reward_pct": 0.5, "props": {}},
            },
            "objects": {
                "altar": {
                    "input_battery.red": 3,
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
                    "input_ore.red": 3,
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
                "attack": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": True},
            },
            "reward_sharing": {
                "groups": {"agent": {"agent": 0.0}, "team_1": {"team_1": 0.5}, "team_2": {"team_2": 0.5}}
            },
            "map_builder": {
                "_target_": "metta.mettagrid.room.multi_room.MultiRoom",
                "num_rooms": 4,
                "border_width": 6,
                "room": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "width": 25,
                    "height": 25,
                    "border_width": 0,
                    "agents": 6,
                    "objects": {"mine": 10, "generator": 2, "altar": 1, "wall": 20, "block": 20},
                },
            },
        },
    }

    # Agent configuration
    agent_config = {
        "_target_": "metta.agent.metta_agent.MettaAgent",
        "hidden_size": 256,
        "rl_layer": {"_target_": "metta.agent.lib.rl_layer.SimpleLSTMRLLayer", "hidden_size": 256, "num_layers": 1},
        "torso": {
            "_target_": "metta.agent.lib.torso.SimpleTorso",
            "hidden_size": 256,
            "resnet_channels": 32,
            "num_resnet_blocks": 2,
            "num_heads": 0,
        },
        "heads": {
            "actor": {
                "_target_": "metta.agent.lib.head.LinearHead",
                "input_size": 256,
                "output_size": "???",
                "num_layers": 0,
            },
            "critic": {
                "_target_": "metta.agent.lib.head.LinearHead",
                "input_size": 256,
                "output_size": 1,
                "num_layers": 0,
            },
        },
        "decoder": {"_target_": "metta.agent.lib.decoder.MultiDiscreteActionDecoder"},
        "sample_dtype": "float32",
        "device": cfg.device,
        "normalize_observations": False,
    }

    # Trainer configuration (based on configs/trainer/puffer.yaml)
    trainer_config = {
        "_target_": "metta.rl.trainer.MettaTrainer",
        "resume": True,
        "use_e3b": False,
        "total_timesteps": getattr(args, "total_timesteps", 1_000_000),
        "clip_coef": 0.1,
        "ent_coef": 0.0021,
        "gae_lambda": 0.916,
        "gamma": 0.977,
        "optimizer": {
            "type": "adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-12,
            "learning_rate": 0.0004573,
            "weight_decay": 0,
        },
        "lr_scheduler": {"enabled": False, "anneal_lr": False},
        "max_grad_norm": 0.5,
        "vf_clip_coef": 0.1,
        "vf_coef": 0.44,
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
        "batch_size": getattr(args, "batch_size", 16384),
        "minibatch_size": 2048,
        "bptt_horizon": 16,
        "update_epochs": 1,
        "cpu_offload": False,
        "compile": False,
        "compile_mode": "reduce-overhead",
        "profiler_interval_epochs": 10000,
        "forward_pass_minibatch_target_size": 2048,
        "async_factor": 2,
        "kickstart": {
            "teacher_uri": None,
            "action_loss_coef": 1,
            "value_loss_coef": 1,
            "anneal_ratio": 0.65,
            "kickstart_steps": 1_000_000_000,
            "additional_teachers": [],
        },
    }

    # Simulation suite configuration for evals
    sim_config = {
        "_target_": "metta.sim.simulation_config.SimulationSuiteConfig",
        "name": "all",
        "num_envs": 32,
        "num_episodes": 10,
        "map_preview_limit": 32,
        "suites": [],
    }

    # Add configurations to main config
    cfg["env"] = DictConfig(env_config)
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
    """Execute training."""
    # Import training dependencies only when needed
    from metta.agent.policy_store import PolicyStore
    from metta.mettagrid.curriculum.core import SingleTaskCurriculum
    from metta.rl.trainer import MettaTrainer

    cfg = build_train_config(args)

    setup_mettagrid_environment(cfg)
    logger = setup_mettagrid_logger("train")

    print(f"Training configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

    # Create output directories
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(f"{cfg.run_dir}/checkpoints", exist_ok=True)

    # Initialize policy store
    policy_store = PolicyStore(cfg, None)

    # Create curriculum
    curriculum = SingleTaskCurriculum("simple_task", cfg.env)

    try:
        # Create trainer
        trainer = MettaTrainer(
            cfg=cfg, wandb_run=None, policy_store=policy_store, sim_suite_config=cfg.train_job.evals, stats_client=None
        )

        # Run training
        trainer.train()
        trainer.close()

        logger.info(f"Training complete. Checkpoints saved to {cfg.run_dir}/checkpoints")
    except KeyError as e:
        if "'groups'" in str(e):
            logger.error("=" * 60)
            logger.error("Configuration Error Detected!")
            logger.error("=" * 60)
            logger.error("")
            logger.error("There's a bug in mettagrid_env.py line 157 where it calls")
            logger.error("cpp_config_dict() on the configuration before passing it to MettaGrid.")
            logger.error("")
            logger.error("The MettaGrid C++ constructor expects the original Python config")
            logger.error("format with 'groups', not the converted format with 'agent_groups'.")
            logger.error("")
            logger.error("To fix this issue:")
            logger.error("1. Edit mettagrid/src/metta/mettagrid/mettagrid_env.py line 157")
            logger.error("2. Change:")
            logger.error("   self._c_env = MettaGrid(cpp_config_dict(game_config_dict), level.grid.tolist())")
            logger.error("3. To:")
            logger.error("   self._c_env = MettaGrid(game_config_dict, level.grid.tolist())")
            logger.error("")
            logger.error("For now, please use tools/train.py with Hydra configuration.")
            logger.error("=" * 60)
            sys.exit(1)
        else:
            raise


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

    print(f"Simulation configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

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

    print(f"Analysis configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

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

    print(f"Dashboard configuration:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

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
  # Train a new policy
  python run.py train --run my_experiment --total-timesteps 1000000

  # Evaluate a policy
  python run.py sim --run my_experiment --policy-uri file://./train_dir/my_experiment/checkpoints

  # Analyze a policy
  python run.py analyze --policy-uri file://./train_dir/my_experiment/checkpoints/policy_v1.pt

  # Generate dashboard
  python run.py dashboard --output-path ./dashboard_data.json

Note: There's currently a known issue with MettaGridEnv initialization outside of Hydra.
      See the error message for details on the fix.
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument("--run", required=True, help="Experiment name")
    train_parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    train_parser.add_argument("--batch-size", type=int, default=16384, help="Batch size")
    train_parser.add_argument("--num-agents", type=int, default=24, help="Number of agents")
    train_parser.add_argument("--device", default="cuda", help="Device to use")
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

    if not args.command:
        parser.print_help()
        sys.exit(1)

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
