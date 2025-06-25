#!/usr/bin/env python3
"""
Standalone training demo for Metta - Alternative to tools/train.py without Hydra

This script shows how to train a Metta agent using explicit configuration
instead of Hydra's YAML-based approach. It follows the same control flow
as tools/train.py but with everything visible and configurable in code.
"""

import os

import numpy as np
import torch
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


def create_simple_env_config():
    """Create a simple environment configuration without Hydra"""
    # Base environment configuration
    env_cfg = {
        "sampling": 0,
        "desync_episodes": False,
        "replay_level_prob": 0.2,
        "game": {
            "num_agents": 24,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            "max_steps": 1000,
            "diversity_bonus": {
                "enabled": False,
                "similarity_coef": 0.5,
                "diversity_coef": 0.5,
            },
            "agent": {
                "default_item_max": 50,
                "heart_max": 255,
                "freeze_duration": 10,
                "rewards": {
                    "action_failure_penalty": 0,
                    "ore.red": 0.005,
                    "ore.blue": 0.005,
                    "ore.green": 0.005,
                    "ore.red_max": 4,
                    "ore.blue_max": 4,
                    "ore.green_max": 4,
                    "battery.red": 0.01,
                    "battery.blue": 0.01,
                    "battery.green": 0.01,
                    "battery.red_max": 5,
                    "battery.blue_max": 5,
                    "battery.green_max": 5,
                    "heart": 1,
                    "heart_max": 1000,
                },
            },
            "groups": {
                "agent": {"id": 0, "sprite": 0, "props": {}},
                "team_1": {"id": 1, "sprite": 1, "group_reward_pct": 0.5, "props": {}},
                "team_2": {"id": 2, "sprite": 4, "group_reward_pct": 0.5, "props": {}},
                "team_3": {"id": 3, "sprite": 8, "group_reward_pct": 0.5, "props": {}},
                "team_4": {"id": 4, "sprite": 1, "group_reward_pct": 0.5, "props": {}},
                "prey": {"id": 5, "sprite": 12, "props": {}},
                "predator": {"id": 6, "sprite": 6, "props": {}},
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
                "mine_blue": {
                    "color": 1,
                    "output_ore.blue": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 50,
                    "initial_items": 1,
                },
                "mine_green": {
                    "output_ore.green": 1,
                    "color": 2,
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
                    "cooldown": 25,
                    "initial_items": 1,
                },
                "generator_blue": {
                    "input_ore.blue": 1,
                    "output_battery.blue": 1,
                    "color": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 25,
                    "initial_items": 1,
                },
                "generator_green": {
                    "input_ore.green": 1,
                    "output_battery.green": 1,
                    "color": 2,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 25,
                    "initial_items": 1,
                },
                "armory": {
                    "input_ore.red": 3,
                    "output_armor": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_items": 1,
                },
                "lasery": {
                    "input_ore.red": 1,
                    "input_battery.red": 2,
                    "output_laser": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_items": 1,
                },
                "lab": {
                    "input_ore.red": 3,
                    "input_battery.red": 3,
                    "output_blueprint": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 5,
                    "initial_items": 1,
                },
                "factory": {
                    "input_blueprint": 1,
                    "input_ore.red": 5,
                    "input_battery.red": 5,
                    "output_armor": 5,
                    "output_laser": 5,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 5,
                    "initial_items": 1,
                },
                "temple": {
                    "input_heart": 1,
                    "input_blueprint": 1,
                    "output_heart": 5,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 5,
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
                "groups": {
                    "team_1": {"team_1": 0.5},
                    "team_2": {"team_2": 0.5},
                    "team_3": {"team_3": 0.5},
                    "team_4": {"team_4": 0.5},
                }
            },
            "map_builder": {
                "_target_": "metta.map.mapgen.MapGen",
                "width": 50,
                "height": 50,
                "border_width": 3,
                "root": {
                    "type": "metta.map.scenes.random.Random",
                    "params": {
                        "objects": {
                            "wall": 20,
                            "block": 20,
                            "altar": 3,
                            "mine_red": 3,
                            "generator_red": 3,
                        },
                        "agents": 24,
                    },
                },
            },
        },
    }

    return DictConfig(env_cfg)


def main():
    print("Starting Metta standalone training demo...")
    print("=" * 60)

    # Register resolvers for config paths
    register_resolvers()

    # Initialize Hydra minimally (required for MettaGridEnv's internal use)
    # This is only needed because MettaGridEnv uses hydra.instantiate internally
    config_path = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_path, version_base=None):
        # =========================================================================
        # Configuration (replaces Hydra YAML files)
        # =========================================================================

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Run configuration
        run_dir = "./demo_run"
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training hyperparameters (from configs/trainer/*)
        total_timesteps = 10_000  # Reduced for demo - was 100_000
        batch_size = 6144
        minibatch_size = 256
        bptt_horizon = 64

        # PPO hyperparameters
        learning_rate = 0.0004573
        gamma = 0.977
        gae_lambda = 0.916
        clip_coef = 0.1
        ent_coef = 0.0021
        vf_coef = 0.44
        max_grad_norm = 0.5
        update_epochs = 1

        # Environment configuration
        num_workers = 1
        async_factor = 2
        forward_pass_minibatch_target_size = 32

        # =========================================================================
        # Create curriculum and environments
        # =========================================================================

        print(f"Training on device: {device}")

        # Create environment configuration directly without Hydra
        env_cfg = create_simple_env_config()

        # Create a simple single-task curriculum
        curriculum = SingleTaskCurriculum("simple_task", env_cfg)

        # Debug: verify the configuration structure
        task = curriculum.get_task()
        game_cfg = task.env_cfg().game
        print(f"Debug - game config keys: {list(game_cfg.keys())}")
        if "groups" in game_cfg:
            print(f"Debug - groups found: {list(game_cfg.groups.keys())}")
        else:
            print("Debug - ERROR: groups not found in game config!")

        # Calculate environment batch sizes
        num_agents = curriculum.get_task().env_cfg().game.num_agents
        target_batch_size = forward_pass_minibatch_target_size // num_agents
        target_batch_size = max(2, target_batch_size)

        env_batch_size = (target_batch_size // num_workers) * num_workers
        env_batch_size = max(num_workers, env_batch_size)
        num_envs = env_batch_size * async_factor

        if num_workers == 1:
            env_batch_size = num_envs

        print(f"Creating {num_envs} environments...")

        # Create vectorized environment
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
        print()

        # =========================================================================
        # Create or load policy
        # =========================================================================

        print("Setting up policy...")

        # Create policy store
        cfg = DictConfig({"device": str(device), "data_dir": run_dir})
        policy_store = PolicyStore(cfg, wandb_run=None)

        # Load checkpoint if it exists
        checkpoint = TrainerCheckpoint.load(run_dir)
        if checkpoint.policy_path:
            policy_record = policy_store.load_from_uri(checkpoint.policy_path)
            print("Loaded existing policy from checkpoint")
        else:
            policy_record = policy_store.create(metta_grid_env)
            print("Created new policy")

        # Get policy and move to device
        policy = policy_record.policy().to(device)

        # Activate actions for this environment
        policy.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, device)

        # =========================================================================
        # Create optimizer and experience buffer
        # =========================================================================

        # Create optimizer
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-12,
            weight_decay=0.0,
        )

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

        # =========================================================================
        # Main training loop
        # =========================================================================

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

        # Main training loop - this is the key pattern!
        while agent_step < total_timesteps:
            steps_before = agent_step

            # ========== ROLLOUT: Collect experience ==========
            with timer("rollout"):
                agent_step, stats = rollout(
                    policy=policy,
                    vecenv=vecenv,
                    experience=experience,
                    device=device,
                    agent_step=agent_step,
                    timer=timer,
                )

            # ========== TRAIN: Update policy with PPO ==========
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

            # ========== LOGGING ==========
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

            # ========== CHECKPOINTING ==========
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

        # =========================================================================
        # Training complete
        # =========================================================================

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
        print(f"   Final model saved to: {path}")
        print(f"   Total steps: {agent_step}")
        print(f"   Total epochs: {epoch}")


if __name__ == "__main__":
    main()
