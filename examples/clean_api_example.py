#!/usr/bin/env python3
"""
Example of using Metta's clean API without Hydra.

This demonstrates how to create and train a Metta agent using the clean,
torch.rl-style API we've built.
"""

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


def main():
    # Import pufferlib for advantage computation
    try:
        from pufferlib import _C  # noqa: F401
    except ImportError:
        raise ImportError("Failed to import pufferlib C extensions. Try installing with --no-build-isolation")

    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment config
    env_config = metta.env(num_agents=2, width=20, height=15, max_steps=1000)

    # Create vectorized environment
    num_envs = 8  # Reduced from 32 to keep total agents reasonable
    num_workers = 1
    batch_size = 256  # This is fine for 16 total agents (8 envs × 2 agents)
    vecenv = metta.make_vecenv(
        env_config=env_config,
        num_envs=num_envs,
        num_workers=num_workers,
        device=str(device),
        vectorization="serial",  # or "multiprocessing", "ray"
    )

    # Get environment info
    env_info = vecenv.driver_env

    # Create observation space (Gym-style)
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

    # Activate actions
    policy.activate_actions(env_info.action_names, env_info.max_action_args, device)

    # Create optimizer
    optimizer = metta.make_optimizer(policy.parameters(), learning_rate=0.0003)

    # Create experience buffer
    experience = metta.make_experience_buffer(
        total_agents=vecenv.num_agents,
        batch_size=batch_size,
        bptt_horizon=8,
        minibatch_size=32,
        max_minibatch_size=32,
        obs_space=env_info.single_observation_space,
        atn_space=env_info.single_action_space,
        device=device,
        hidden_size=policy.hidden_size,
        num_lstm_layers=policy.core_num_layers,
    )

    # Create loss module (torch.rl style!)
    loss_module = metta.make_loss_module(policy=policy, vf_coef=0.5, ent_coef=0.01, clip_coef=0.2)

    # Create losses tracker
    losses = Losses()

    # Training hyperparameters
    total_timesteps = 1_000_000
    gamma = 0.99
    gae_lambda = 0.95

    # Create timer
    logger = metta.get_logger("example")
    timer = Stopwatch(logger)
    timer.start()

    print("Starting training...")
    vecenv.async_reset(seed=42)

    agent_step = 0

    # Run training for 5 epochs
    for epoch in range(5):
        raw_infos = []
        experience.reset_for_rollout()

        # Collect rollout
        while not experience.ready_for_training:
            num_steps, info, _ = perform_rollout_step(policy, vecenv, experience, device, timer)
            agent_step += num_steps
            if info:
                raw_infos.extend(info)

        rollout_stats = process_rollout_infos(raw_infos)

        # Training phase
        losses.zero()
        experience.reset_importance_sampling_ratios()

        # Compute advantages
        advantages = compute_initial_advantages(experience, gamma, gae_lambda, 1.0, 1.0, device)

        # Train on minibatches
        for minibatch_idx in range(experience.num_minibatches):
            minibatch = experience.sample_minibatch(
                advantages=advantages,
                prio_alpha=0.0,
                prio_beta=0.6,
                minibatch_idx=minibatch_idx,
                total_minibatches=experience.num_minibatches,
            )

            # Compute loss using our torch.rl-style module
            loss = loss_module(
                minibatch=minibatch,
                experience=experience,
                losses=losses,
                agent_step=agent_step,
                device=device,
            )

            # Standard PyTorch training
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            losses.minibatches_processed += 1

        # Log progress
        loss_stats = losses.stats()
        print(
            f"Epoch {epoch} - Policy loss: {loss_stats['policy_loss']:.4f} - Value loss: {loss_stats['value_loss']:.4f}"
        )

    print("Training complete!")

    # Save the trained policy
    torch.save(policy.state_dict(), "trained_policy.pt")

    # Clean up
    vecenv.close()


if __name__ == "__main__":
    main()
