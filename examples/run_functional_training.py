#!/usr/bin/env python3
"""Example: How to run training with the new functional approach."""

import torch

from metta.agent import create_agent
from metta.agent.policy_store import MemoryPolicyStore
from metta.rl.configs import PPOConfig, TrainerConfig
from metta.rl.functional_trainer import functional_training_loop
from metta.train.train_config import TrainingConfig, small_fast_config
from mettagrid import create_env
from mettagrid.curriculum import SimpleCurriculum


def main():
    """Run training using the new functional approach."""

    # Method 1: Using structured configs (recommended)
    print("=== Method 1: Using Structured Configs ===")

    # Create a complete training config
    config = TrainingConfig(
        experiment_name="my_functional_experiment",
        agent=TrainingConfig.agent.__class__(
            name="simple_cnn",
            hidden_size=256,
        ),
        environment=TrainingConfig.environment.__class__(
            width=15,
            height=15,
            max_steps=500,
        ),
        trainer=TrainerConfig(
            total_timesteps=100_000,
            batch_size=2048,
            minibatch_size=256,
            checkpoint_interval=100,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        ppo=PPOConfig(
            clip_coef=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        ),
    )

    # Create environment
    env = create_env(
        width=config.environment.width,
        height=config.environment.height,
        max_steps=config.environment.max_steps,
    )

    # Create agent
    agent = create_agent(
        agent_name=config.agent.name,
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        hidden_size=config.agent.hidden_size,
        device=config.trainer.device,
    )

    # Create curriculum
    curriculum = SimpleCurriculum(
        base_env_config={
            "width": config.environment.width,
            "height": config.environment.height,
            "max_steps": config.environment.max_steps,
        }
    )

    # Create policy store
    policy_store = MemoryPolicyStore()

    # Run training
    print(f"Starting training: {config.experiment_name}")
    print(f"Agent: {config.agent.name}, Environment: {config.environment.width}x{config.environment.height}")
    print(f"Total timesteps: {config.trainer.total_timesteps}")

    state = functional_training_loop(
        config=config.trainer,
        ppo_config=config.ppo,
        policy=agent,
        curriculum=curriculum,
        policy_store=policy_store,
    )

    print("\nTraining completed!")
    print(f"Final epoch: {state.epoch}")
    print(f"Total steps: {state.agent_steps}")


def quick_start():
    """Quick start with minimal configuration."""
    print("\n=== Method 2: Quick Start ===")

    # Use a preset config
    config = small_fast_config()

    # Create minimal components
    env = create_env(width=10, height=10, max_steps=100)
    agent = create_agent("simple_cnn", env.observation_space, env.action_space)
    curriculum = SimpleCurriculum(base_env_config={"width": 10, "height": 10})
    policy_store = MemoryPolicyStore()

    # Run training
    state = functional_training_loop(
        config=config.trainer,
        ppo_config=config.ppo,
        policy=agent,
        curriculum=curriculum,
        policy_store=policy_store,
    )

    print(f"Quick training completed in {state.epoch} epochs!")


def direct_components():
    """Use components directly for maximum control."""
    print("\n=== Method 3: Direct Component Usage ===")

    from metta.rl.collectors import RolloutCollector
    from metta.rl.experience import Experience
    from metta.rl.optimizers import PPOOptimizer
    from metta.rl.vecenv import make_vecenv

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = create_env(width=10, height=10)
    agent = create_agent("simple_cnn", env.observation_space, env.action_space, device=device)

    # Create vectorized environment
    curriculum = SimpleCurriculum(base_env_config={"width": 10, "height": 10})
    vecenv = make_vecenv(
        curriculum,
        "serial",
        num_envs=4,
        batch_size=4,
        num_workers=1,
    )

    # Create experience buffer
    experience = Experience(
        total_agents=vecenv.num_agents,
        batch_size=1024,
        bptt_horizon=16,
        minibatch_size=128,
        obs_space=vecenv.single_observation_space,
        atn_space=vecenv.single_action_space,
        device=device,
        hidden_size=256,
    )

    # Create components
    collector = RolloutCollector(vecenv, agent, experience, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    ppo = PPOOptimizer(agent, optimizer, device, PPOConfig())

    # Training loop
    print("Running 10 training iterations...")
    for i in range(10):
        # Collect experience
        stats, steps = collector.collect()
        print(f"Iteration {i + 1}: Collected {steps} steps")

        # Update policy
        losses = ppo.update(experience, update_epochs=4)
        print(f"  Policy loss: {losses['policy_loss']:.4f}")

    vecenv.close()
    print("Direct training completed!")


def custom_training_loop():
    """Example with custom training step and losses."""
    print("\n=== Method 4: Custom Training Loop ===")

    from metta.rl.functional_trainer import (
        default_training_step,
    )

    # Custom loss function
    def exploration_bonus_loss(policy, obs, actions, rewards, values, advantages):
        """Add exploration bonus based on action diversity."""
        # Simple entropy bonus (already in PPO, but this is an example)
        action_probs = torch.softmax(policy(obs, None)[0], dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        return -0.01 * entropy.mean()  # Negative because we want to maximize entropy

    # Custom training step
    def my_training_step(state, components, config, experience, custom_losses=None):
        """Custom training step with additional logging."""
        # Run default training
        metrics = default_training_step(state, components, config, experience, custom_losses)

        # Add custom logic
        if state.epoch % 5 == 0:
            print(f"[Custom] Epoch {state.epoch}: loss={metrics.get('policy_loss', 0):.4f}")

        # Early stopping
        if metrics.get("policy_loss", float("inf")) < 0.1:
            print("[Custom] Reached target loss, stopping early!")
            state.should_stop = True

        return metrics

    # Setup components
    config = small_fast_config()
    env = create_env(width=10, height=10)
    agent = create_agent("simple_cnn", env.observation_space, env.action_space)
    curriculum = SimpleCurriculum(base_env_config={"width": 10, "height": 10})
    policy_store = MemoryPolicyStore()

    # Run with custom components
    state = functional_training_loop(
        config=config.trainer,
        ppo_config=config.ppo,
        policy=agent,
        curriculum=curriculum,
        policy_store=policy_store,
        step_fn=my_training_step,
        custom_losses=[exploration_bonus_loss],
    )

    print(f"Custom training completed in {state.epoch} epochs!")


if __name__ == "__main__":
    print("Metta Functional Training Examples\n")

    # Run different examples
    try:
        # 1. Full structured config example
        main()

        # 2. Quick start
        quick_start()

        # 3. Direct component usage
        direct_components()

        # 4. Custom training loop
        custom_training_loop()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
