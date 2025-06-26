#!/usr/bin/env python3
"""
Standalone training demo for Metta using YAML configuration

This script shows how to train a Metta agent using a comprehensive YAML
configuration file instead of Hydra's complex defaults system.
"""

import os

import numpy as np
import torch
import yaml
from hydra import initialize_config_dir
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.common.stopwatch import Stopwatch
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functional_trainer import rollout, train_ppo
from metta.rl.losses import Losses
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.vecenv import make_vecenv
from metta.util.resolvers import register_resolvers


def main():
    print("Metta Training Demo with YAML Configuration")
    print("=" * 60)

    # Register resolvers for config paths
    register_resolvers()

    # Load configuration from YAML file
    print("\nLoading configuration from configs/env/common.yaml...")
    with open("configs/env/common.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    # Convert to DictConfig for compatibility
    env_cfg = DictConfig(env_config)

    # Initialize Hydra minimally (required for MettaGridEnv's internal use of instantiate)
    config_path = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_path, version_base=None):
        # Basic setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")

        # Run configuration
        run_dir = "./demo_run"
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training hyperparameters
        total_timesteps = 100_000
        batch_size = 6144
        minibatch_size = 256
        bptt_horizon = 64

        # PPO hyperparameters
        learning_rate = 0.0003
        gamma = 0.99
        gae_lambda = 0.95
        clip_coef = 0.2
        ent_coef = 0.01
        vf_coef = 0.5
        max_grad_norm = 0.5
        update_epochs = 4

        # Environment configuration
        num_workers = 1
        async_factor = 2
        forward_pass_minibatch_target_size = 32

        # Create curriculum from loaded config
        print("\nCreating environment from YAML configuration...")
        curriculum = SingleTaskCurriculum("yaml_task", env_cfg)

        # Calculate environment batch sizes
        num_agents = curriculum.get_task().env_cfg().game.num_agents
        target_batch_size = forward_pass_minibatch_target_size // num_agents
        target_batch_size = max(2, target_batch_size)

        env_batch_size = (target_batch_size // num_workers) * num_workers
        env_batch_size = max(num_workers, env_batch_size)
        num_envs = env_batch_size * async_factor

        if num_workers == 1:
            env_batch_size = num_envs

        print(f"Creating {num_envs} environments with {num_agents} agents each...")

        # Create vectorized environment
        try:
            vecenv = make_vecenv(
                curriculum,
                "serial",
                num_envs=num_envs,
                batch_size=env_batch_size,
                num_workers=num_workers,
                zero_copy=True,
            )

            # Reset environments
            seed = np.random.randint(0, 1000000)
            vecenv.async_reset(seed)

            # Get environment interface
            metta_grid_env: MettaGridEnv = vecenv.driver_env

            print(f"✓ Successfully created {num_envs} environments")
            print(f"  Environment type: {type(metta_grid_env).__name__}")
            print(f"  Observation space: {metta_grid_env.single_observation_space}")
            print(f"  Action space: {metta_grid_env.single_action_space}")
            print(f"  Action names: {metta_grid_env.action_names}")
            print()

        except Exception as e:
            print(f"\n❌ Error creating environment: {type(e).__name__}: {e}")
            print("\nThis typically happens due to configuration issues.")
            print("Check that configs/env/common.yaml has all required fields.")
            raise

        # Create policy store
        cfg = DictConfig({"device": str(device), "data_dir": run_dir})
        policy_store = PolicyStore(cfg, wandb_run=None)

        # Load or create policy
        print("Setting up policy...")
        checkpoint = TrainerCheckpoint.load(run_dir)
        if checkpoint.policy_path:
            policy_record = policy_store.load_from_uri(checkpoint.policy_path)
            print("Loaded existing policy from checkpoint")
        else:
            policy_record = policy_store.create(metta_grid_env)
            print("Created new policy")

        # Get policy and move to device
        policy = policy_record.policy().to(device)
        policy.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, device)

        # Create optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        # Restore optimizer state if checkpoint exists
        if checkpoint.agent_step > 0 and checkpoint.optimizer_state_dict:
            try:
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                print("Loaded optimizer state from checkpoint")
            except:
                print("Could not load optimizer state, starting fresh")

        # Create experience buffer
        experience = Experience(
            total_agents=vecenv.num_agents,
            batch_size=batch_size,
            bptt_horizon=bptt_horizon,
            minibatch_size=minibatch_size,
            max_minibatch_size=minibatch_size,
            obs_space=vecenv.single_observation_space,
            atn_space=vecenv.single_action_space,
            device=device,
            hidden_size=getattr(policy, "hidden_size", 256),
            cpu_offload=False,
            num_lstm_layers=2,
            agents_per_batch=getattr(vecenv, "agents_per_batch", None),
        )

        # Initialize training state
        agent_step = checkpoint.agent_step
        epoch = checkpoint.epoch
        losses = Losses()
        timer = Stopwatch(None)
        timer.start()

        # Config object for train_ppo
        agent_cfg = DictConfig({"agent": {"clip_range": 0}})

        print(f"\nStarting training from epoch {epoch}, step {agent_step}")
        print("=" * 60)

        # Main training loop
        while agent_step < total_timesteps:
            steps_before = agent_step

            # ROLLOUT: Collect experience
            with timer("rollout"):
                agent_step, stats = rollout(
                    policy=policy,
                    vecenv=vecenv,
                    experience=experience,
                    device=device,
                    agent_step=agent_step,
                    timer=timer,
                )

            # TRAIN: Update policy with PPO
            with timer("train"):
                epoch = train_ppo(
                    policy=policy,
                    optimizer=optimizer,
                    experience=experience,
                    device=device,
                    losses=losses,
                    epoch=epoch,
                    cfg=agent_cfg,
                    lr_scheduler=None,
                    timer=timer,
                    # PPO hyperparameters
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_coef=clip_coef,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    norm_adv=True,
                    clip_vloss=True,
                    vf_clip_coef=0.1,
                    update_epochs=update_epochs,
                    target_kl=None,
                    kickstarter=None,
                    agent_step=agent_step,
                    l2_reg_loss_coef=0.0,
                    l2_init_loss_coef=0.0,
                    clip_range=0,
                    # Prioritized replay
                    prio_alpha=0.0,
                    prio_beta0=0.6,
                    total_timesteps=total_timesteps,
                    batch_size=batch_size,
                    # V-trace
                    vtrace_rho_clip=1.0,
                    vtrace_c_clip=1.0,
                )

            # Log progress
            rollout_time = timer.get_last_elapsed("rollout")
            train_time = timer.get_last_elapsed("train")
            total_time = train_time + rollout_time
            steps_calculated = agent_step - steps_before
            steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

            loss_stats = losses.stats()

            print(
                f"Epoch {epoch:4d} | "
                f"Steps: {agent_step:6d}/{total_timesteps} | "
                f"SPS: {steps_per_sec:5.0f} | "
                f"Loss: {loss_stats.get('policy_loss', 0):.4f} | "
                f"Value: {loss_stats.get('value_loss', 0):.4f}"
            )

            # Checkpointing
            if epoch % 10 == 0:
                print(f"Saving checkpoint at epoch {epoch}...")

                # Save policy
                name = policy_store.make_model_name(epoch)
                path = os.path.join(checkpoint_dir, name)
                pr = policy_store.save(
                    name=name, path=path, policy=policy, metadata={"epoch": epoch, "agent_step": agent_step}
                )

                # Save training state
                checkpoint = TrainerCheckpoint(
                    agent_step=agent_step,
                    epoch=epoch,
                    total_agent_step=agent_step,
                    optimizer_state_dict=optimizer.state_dict(),
                    policy_path=pr.uri if pr else None,
                    extra_args={},
                )
                checkpoint.save(run_dir)

        # Final checkpoint
        print("\nTraining complete! Saving final checkpoint...")
        name = policy_store.make_model_name(epoch)
        path = os.path.join(checkpoint_dir, name)
        pr = policy_store.save(
            name=name, path=path, policy=policy, metadata={"epoch": epoch, "agent_step": agent_step, "final": True}
        )

        # Clean up
        vecenv.close()

        print("\n✅ Training finished!")
        print("   Configuration loaded from: configs/env/common.yaml")
        print(f"   Final model saved to: {path}")
        print(f"   Total steps: {agent_step}")
        print(f"   Total epochs: {epoch}")


if __name__ == "__main__":
    main()
