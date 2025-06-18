#!/usr/bin/env python
"""Simplified training script using functional API.

This script demonstrates the new pythonic approach to training,
where we create objects directly and use functional training components.
"""

import argparse
import time

import wandb

# Import agent and environment creation
from metta.agent import MettaAgent
from metta.rl.experience import Experience
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation_config import SimulationSuiteConfig

# Import our functional training API
from metta.train import (
    OptimizerConfig,
    PPOLossConfig,
    RolloutConfig,
    compute_ppo_loss,
    evaluate_policy,
    find_latest_checkpoint,
    load_checkpoint,
    rollout,
    save_checkpoint,
    update_policy,
)
from metta.train.update import create_lr_scheduler, create_optimizer
from mettagrid.curriculum import curriculum_from_config_path
from mettagrid.mettagrid_env import MettaGridEnv


def create_agent(env: MettaGridEnv, config: dict) -> MettaAgent:
    """Create a MettaAgent directly in Python."""
    # Get environment info
    obs_shape = env.single_observation_space.shape
    action_names = env.action_names
    max_action_args = env.max_action_args

    # Create agent with sensible defaults
    agent = MettaAgent(
        observations={"obs_key": "grid_obs"},
        clip_range=0,
        analyze_weights_interval=300,
        components={
            # Core components defined in Python
            "_obs_": {
                "_target_": "metta.agent.lib.obs_shaper.ObsShaper",
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
                    "out_channels": 64,
                    "kernel_size": 5,
                    "stride": 3,
                },
            },
            "cnn2": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "cnn1"}],
                "nn_params": {
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                },
            },
            "obs_flattener": {
                "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                "sources": [{"name": "cnn2"}],
            },
            "fc1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "obs_flattener"}],
                "nn_params": {"out_features": 512},
            },
            "_core_": {
                "_target_": "metta.agent.lib.lstm.LSTM",
                "sources": [{"name": "fc1"}],
                "output_size": 512,
                "nn_params": {"num_layers": 2},
            },
            "critic_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 1024},
            },
            "_value_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "critic_1"}],
                "nn_params": {"out_features": 1},
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
    )

    # Activate actions for the environment
    agent.activate_actions(action_names, max_action_args, config["device"])

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train a Metta agent")
    parser.add_argument("--run-name", type=str, required=True, help="Name for this training run")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--num-envs", type=int, default=128, help="Number of parallel environments")
    parser.add_argument("--batch-size", type=int, default=262144, help="Batch size for training")
    parser.add_argument("--minibatch-size", type=int, default=16384, help="Minibatch size for updates")
    parser.add_argument("--rollout-length", type=int, default=128, help="Length of rollouts")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint every N epochs")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N epochs")
    parser.add_argument("--wandb-project", type=str, default="metta", help="Wandb project name")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    # Configuration
    config = {
        "device": args.device,
        "run_name": args.run_name,
        "run_dir": f"./train_dir/{args.run_name}",
        "checkpoint_dir": f"./train_dir/{args.run_name}/checkpoints",
    }

    # Initialize wandb
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
    )

    # Create environment
    print("🌍 Creating environment...")
    env_config = {
        "game": {
            "num_agents": 1,
            "max_steps": 1000,
            "map_builder": {
                "_target_": "mettagrid.room.mean_distance.MeanDistance",
                "width": 35,
                "height": 35,
                "mean_distance": 25,
                "border_width": 3,
                "agents": 1,
                "objects": {"altar": 3, "wall": 12},
            },
        },
    }

    curriculum = curriculum_from_config_path(env_config, {})
    driver_env = MettaGridEnv(curriculum, render_mode=None)

    # Create vectorized environment
    vecenv = make_vecenv(
        driver_env,
        num_envs=args.num_envs,
        num_workers=4,
        batch_size=args.batch_size,
        rollout_length=args.rollout_length,
    )

    # Create agent
    print("🤖 Creating agent...")
    agent = create_agent(driver_env, config).to(args.device)

    # Create experience buffer
    experience = Experience(
        total_agents=args.num_envs,
        batch_size=args.batch_size,
        bptt_horizon=args.rollout_length,
        minibatch_size=args.minibatch_size,
        max_minibatch_size=args.minibatch_size,
        obs_space=driver_env.single_observation_space,
        atn_space=driver_env.single_action_space,
        device=args.device,
        use_rnn=True,
        hidden_size=512,
        cpu_offload=False,
    )

    # Create optimizer
    opt_config = OptimizerConfig(
        type="adam",
        learning_rate=args.learning_rate,
        max_grad_norm=0.5,
    )
    optimizer = create_optimizer(agent, opt_config)

    # Create LR scheduler
    total_steps = args.total_timesteps // args.batch_size
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)

    # Load checkpoint if resuming
    epoch = 0
    agent_step = 0
    if args.resume:
        checkpoint_path = find_latest_checkpoint(config["checkpoint_dir"])
        if checkpoint_path:
            print(f"📂 Loading checkpoint from {checkpoint_path}")
            ckpt_info = load_checkpoint(checkpoint_path, agent, optimizer, lr_scheduler, args.device)
            epoch = ckpt_info["epoch"]
            agent_step = ckpt_info["agent_step"]

    # Training configuration
    rollout_config = RolloutConfig(
        num_steps=args.rollout_length,
        device=args.device,
        cpu_offload=False,
    )

    ppo_config = PPOLossConfig(
        clip_coef=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Evaluation configuration
    eval_config = SimulationSuiteConfig(
        {
            "name": "navigation_eval",
            "simulations": {
                "navigation/simple": {
                    "env": "env/mettagrid/navigation/evals/emptyspace_withinsight",
                    "num_episodes": 10,
                },
                "navigation/walls": {
                    "env": "env/mettagrid/navigation/evals/walls_sparse",
                    "num_episodes": 10,
                },
            },
        }
    )

    # Training loop
    print("🚀 Starting training...")
    while agent_step < args.total_timesteps:
        epoch_start_time = time.time()

        # Collect rollout
        rollout_stats = rollout(agent, vecenv, experience, rollout_config)

        # Compute PPO loss and update
        update_epochs = 4
        for _ in range(update_epochs):
            loss, losses = compute_ppo_loss(agent, experience, ppo_config)
            update_policy(agent, optimizer, loss, opt_config, lr_scheduler)

        # Update counters
        epoch += 1
        agent_step += args.batch_size

        # Log stats
        epoch_time = time.time() - epoch_start_time
        steps_per_sec = args.batch_size / epoch_time

        log_dict = {
            "train/agent_step": agent_step,
            "train/epoch": epoch,
            "train/steps_per_sec": steps_per_sec,
            **{f"losses/{k}": v for k, v in losses.to_dict().items()},
            **{f"rollout/{k}": np.mean(v) for k, v in rollout_stats.items() if isinstance(v, list)},
        }
        wandb.log(log_dict)

        print(
            f"Epoch {epoch} | Steps: {agent_step:,} | {steps_per_sec:.0f} steps/sec | "
            f"Loss: {losses.policy_loss:.4f} | Value: {losses.value_loss:.4f}"
        )

        # Checkpoint
        if epoch % args.checkpoint_interval == 0:
            print("💾 Saving checkpoint...")
            save_checkpoint(
                config["checkpoint_dir"],
                agent,
                optimizer,
                epoch,
                agent_step,
                lr_scheduler,
                metadata={"wandb_run_id": wandb_run.id},
            )

        # Evaluate
        if epoch % args.eval_interval == 0:
            print("📊 Evaluating...")
            eval_result = evaluate_policy(agent, eval_config, args.device)
            print(f"  Overall score: {eval_result.overall_score:.3f}")
            print(f"  Category scores: {eval_result.category_scores}")

            wandb.log(
                {
                    "eval/overall_score": eval_result.overall_score,
                    **{f"eval/{cat}_score": score for cat, score in eval_result.category_scores.items()},
                }
            )

    # Final checkpoint
    save_checkpoint(
        config["checkpoint_dir"],
        agent,
        optimizer,
        epoch,
        agent_step,
        lr_scheduler,
        metadata={"wandb_run_id": wandb_run.id, "final": True},
    )

    print("✅ Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
