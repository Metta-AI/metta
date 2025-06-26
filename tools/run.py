#!/usr/bin/env python3
"""
Clean, simple entry point for Metta without Hydra.

Usage:
    # Train (default)
    python run.py

    # Train with options
    python run.py train --run my_experiment --timesteps 1000000

    # Evaluate
    python run.py sim --run my_experiment --policy-uri file://./checkpoints/policy.pt
"""

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

import metta_api as metta
from metta.common.stopwatch import Stopwatch
from metta.rl.functional_trainer import (
    compute_initial_advantages,
    perform_rollout_step,
    process_rollout_infos,
)
from metta.rl.losses import Losses


def train(args):
    """Train a Metta agent."""
    # Setup
    config = metta.build_runtime_config(
        run=args.run,
        device=args.device,
        seed=args.seed,
        vectorization=args.vectorization,
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("train")

    # Training parameters
    device = torch.device(config["device"])
    checkpoint_dir = f"{config['run_dir']}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Training {args.run}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Timesteps: {args.timesteps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Checkpoint dir: {checkpoint_dir}")

    # Import pufferlib for advantage computation
    try:
        from pufferlib import _C  # noqa: F401
    except ImportError:
        raise ImportError("Failed to import pufferlib C extensions. Try installing with --no-build-isolation")

    # Create environment
    env_config = metta.env(num_agents=args.num_agents)

    # Calculate environment count
    target_batch_size = 32 // args.num_agents
    if target_batch_size < args.num_workers:
        target_batch_size = args.num_workers
    forward_batch_size = (target_batch_size // args.num_workers) * args.num_workers
    num_envs = forward_batch_size

    # Create vectorized environment
    vecenv = metta.make_vecenv(
        env_config=env_config,
        num_envs=num_envs,
        num_workers=args.num_workers,
        batch_size=None if args.num_workers == 1 else forward_batch_size,
        device=str(device),
        vectorization=args.vectorization,
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
    policy = metta.make_agent(
        obs_space=obs_space,
        action_space=env_info.single_action_space,
        obs_width=env_info.obs_width,
        obs_height=env_info.obs_height,
        feature_normalizations=env_info.feature_normalizations,
        global_features=env_info.global_features,
        device=device,
    )
    policy.activate_actions(env_info.action_names, env_info.max_action_args, device)

    # Create experience buffer
    minibatch_size = min(32, args.batch_size)
    while args.batch_size % minibatch_size != 0 and minibatch_size > 1:
        minibatch_size -= 1

    experience = metta.make_experience_buffer(
        total_agents=vecenv.num_agents,
        batch_size=args.batch_size,
        bptt_horizon=8,
        minibatch_size=minibatch_size,
        max_minibatch_size=32,
        obs_space=env_info.single_observation_space,
        atn_space=env_info.single_action_space,
        device=device,
        hidden_size=policy.hidden_size,
        num_lstm_layers=policy.core_num_layers,
        agents_per_batch=getattr(vecenv, "agents_per_batch", None),
    )

    # Create optimizer and loss module
    optimizer = metta.make_optimizer(policy.parameters(), learning_rate=args.learning_rate)
    loss_module = metta.make_loss_module(policy=policy)
    losses = Losses()

    # Training setup
    timer = Stopwatch(logger)
    timer.start()

    logger.info("Starting training...")
    vecenv.async_reset(seed=config["seed"])

    agent_step = 0
    epoch = 0

    # Training loop
    while agent_step < args.timesteps:
        steps_before = agent_step

        # Rollout
        with timer("rollout"):
            raw_infos = []
            experience.reset_for_rollout()

            while not experience.ready_for_training:
                num_steps, info, _ = perform_rollout_step(policy, vecenv, experience, device, timer)
                agent_step += num_steps
                if info:
                    raw_infos.extend(info)

            rollout_stats = process_rollout_infos(raw_infos)

        # Train
        with timer("train"):
            losses.zero()
            experience.reset_importance_sampling_ratios()

            # Compute advantages
            advantages = compute_initial_advantages(experience, 0.977, 0.916, 1.0, 1.0, device)

            # Train minibatches
            for mb_idx in range(experience.num_minibatches):
                minibatch = experience.sample_minibatch(
                    advantages=advantages,
                    prio_alpha=0.0,
                    prio_beta=0.6,
                    minibatch_idx=mb_idx,
                    total_minibatches=experience.num_minibatches,
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

                if (mb_idx + 1) % experience.accumulate_minibatches == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
                    if hasattr(policy, "clip_weights"):
                        policy.clip_weights()

        # Log progress
        steps_in_epoch = agent_step - steps_before
        rollout_time = timer.get_last_elapsed("rollout")
        train_time = timer.get_last_elapsed("train")
        steps_per_sec = steps_in_epoch / (rollout_time + train_time)

        loss_stats = losses.stats()
        logger.info(
            f"Epoch {epoch} - Steps: {agent_step}/{args.timesteps} - "
            f"{steps_per_sec:.0f} sps - "
            f"Policy loss: {loss_stats['policy_loss']:.4f} - "
            f"Value loss: {loss_stats['value_loss']:.4f}"
        )

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/policy_epoch_{epoch}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        epoch += 1

    # Save final checkpoint
    final_checkpoint = f"{checkpoint_dir}/policy_final.pt"
    torch.save(policy.state_dict(), final_checkpoint)
    logger.info(f"Training complete! Final checkpoint: {final_checkpoint}")

    vecenv.close()


def sim(args):
    """Evaluate a trained policy."""
    from metta.agent.policy_store import PolicyStore
    from metta.sim.simulation_config import SimulationSuiteConfig
    from metta.sim.simulation_suite import SimulationSuite

    # Setup
    config = metta.build_runtime_config(
        run=args.run,
        device=args.device,
        vectorization="multiprocessing",  # Better for eval
    )
    metta.setup_metta_environment(config)
    logger = metta.get_logger("sim")

    # Directories
    stats_dir = f"{config['run_dir']}/stats"
    replay_dir = f"{config['run_dir']}/replays/evals"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    logger.info(f"Evaluating policy: {args.policy_uri}")

    # Load policy
    policy_store = PolicyStore(config, None)
    policy_prs = policy_store.policies(args.policy_uri, "top", n=1, metric="eval_score")

    results = {"policies": []}

    for pr in policy_prs:
        logger.info(f"Evaluating {pr.name}")

        # Run simulation
        sim = SimulationSuite(
            config=SimulationSuiteConfig(
                {
                    "name": "eval",
                    "num_envs": args.num_envs,
                    "num_episodes": args.num_episodes,
                    "map_preview_limit": 32,
                    "suites": [],
                }
            ),
            policy_pr=pr,
            policy_store=policy_store,
            replay_dir=f"{replay_dir}/{pr.name}",
            stats_dir=stats_dir,
            device=config["device"],
            vectorization=config["vectorization"],
            stats_client=None,
        )

        sim_results = sim.simulate()

        # Collect results
        checkpoint_data = {"name": pr.name, "uri": pr.uri, "metrics": {}}

        rewards_df = sim_results.stats_db.query(
            "SELECT AVG(value) AS reward_avg FROM agent_metrics WHERE metric = 'reward'"
        )
        if len(rewards_df) > 0 and rewards_df.iloc[0]["reward_avg"] is not None:
            checkpoint_data["metrics"]["reward_avg"] = float(rewards_df.iloc[0]["reward_avg"])

        results["policies"].append(checkpoint_data)

        # Export stats
        stats_db_uri = f"{config['run_dir']}/stats.db"
        sim_results.stats_db.export(stats_db_uri)
        logger.info(f"Exported stats to {stats_db_uri}")

    # Print results
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Metta - Clean RL training without Hydra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument("--run", default="default_run", help="Experiment name")
    train_parser.add_argument("--timesteps", type=int, default=10_000, help="Total timesteps")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
    train_parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    train_parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")
    train_parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint interval")
    train_parser.add_argument("--device", help="Device (defaults to cuda if available)")
    train_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    train_parser.add_argument("--vectorization", default="serial", help="Vectorization mode")

    # Sim command
    sim_parser = subparsers.add_parser("sim", help="Evaluate a policy")
    sim_parser.add_argument("--run", required=True, help="Experiment name")
    sim_parser.add_argument("--policy-uri", required=True, help="Policy URI")
    sim_parser.add_argument("--num-envs", type=int, default=32, help="Number of environments")
    sim_parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    sim_parser.add_argument("--device", help="Device (defaults to cuda if available)")

    args = parser.parse_args()

    # Default to train if no command
    if not args.command:
        args.command = "train"
        args.run = "default_run"
        args.timesteps = 10_000
        args.batch_size = 256
        args.learning_rate = 0.0003
        args.num_agents = 2
        args.num_workers = 1
        args.checkpoint_interval = 100
        args.device = None
        args.seed = 0
        args.vectorization = "serial"

    # Execute command
    if args.command == "train":
        train(args)
    elif args.command == "sim":
        sim(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
