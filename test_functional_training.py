#!/usr/bin/env python3
"""Quick test to verify functional training works."""

import torch

from metta.agent import create_agent
from metta.agent.policy_store import MemoryPolicyStore
from metta.rl.functional_trainer import functional_training_loop
from metta.train.train_config import PPOConfig, TrainerConfig
from mettagrid import create_env
from mettagrid.curriculum import SimpleCurriculum


def test_basic_training():
    """Test that basic training runs without errors."""
    print("Testing basic functional training...")

    # Create minimal config for fast testing
    trainer_config = TrainerConfig(
        total_timesteps=1000,  # Very short for testing
        batch_size=256,
        minibatch_size=64,
        checkpoint_interval=1000,  # Don't checkpoint during test
        device="cpu",  # Use CPU for testing
    )

    ppo_config = PPOConfig(
        update_epochs=2,  # Fewer epochs for speed
    )

    # Create small environment
    env = create_env(width=5, height=5, max_steps=50)

    # Create agent
    agent = create_agent(
        "simple_cnn",
        env.observation_space,
        env.action_space,
        hidden_size=32,  # Small for testing
        device="cpu",
    )

    # Create curriculum
    curriculum = SimpleCurriculum(
        base_env_config={
            "width": 5,
            "height": 5,
            "max_steps": 50,
        }
    )

    # Create policy store
    policy_store = MemoryPolicyStore()

    # Run training
    print("Starting training loop...")
    state = functional_training_loop(
        config=trainer_config,
        ppo_config=ppo_config,
        policy=agent,
        curriculum=curriculum,
        policy_store=policy_store,
    )

    print("✓ Training completed successfully!")
    print(f"  - Final epoch: {state.epoch}")
    print(f"  - Total steps: {state.agent_steps}")

    # Basic assertions
    assert state.agent_steps >= trainer_config.total_timesteps, "Should have trained for at least total_timesteps"
    assert state.epoch > 0, "Should have completed at least one epoch"

    return True


def test_custom_loss():
    """Test that custom losses work."""
    print("\nTesting custom loss function...")

    # Simple custom loss that just adds L2 regularization
    def l2_regularization(policy, obs, actions, rewards, values, advantages):
        """Add L2 regularization to policy weights."""
        l2_sum = 0
        for param in policy.parameters():
            l2_sum += torch.sum(param**2)
        return 0.0001 * l2_sum

    trainer_config = TrainerConfig(
        total_timesteps=500,
        batch_size=128,
        minibatch_size=32,
        device="cpu",
    )

    ppo_config = PPOConfig(update_epochs=1)

    env = create_env(width=5, height=5)
    agent = create_agent("simple_cnn", env.observation_space, env.action_space, hidden_size=32, device="cpu")
    curriculum = SimpleCurriculum(base_env_config={"width": 5, "height": 5})
    policy_store = MemoryPolicyStore()

    # Run with custom loss
    state = functional_training_loop(
        config=trainer_config,
        ppo_config=ppo_config,
        policy=agent,
        curriculum=curriculum,
        policy_store=policy_store,
        custom_losses=[l2_regularization],
    )

    print("✓ Custom loss training completed!")
    print(f"  - Steps: {state.agent_steps}")

    return True


def test_direct_components():
    """Test using components directly."""
    print("\nTesting direct component usage...")

    from metta.rl.collectors import RolloutCollector
    from metta.rl.experience import Experience
    from metta.rl.optimizers import PPOOptimizer
    from metta.rl.vecenv import make_vecenv

    device = torch.device("cpu")

    # Create environment
    curriculum = SimpleCurriculum(base_env_config={"width": 5, "height": 5, "max_steps": 50})
    vecenv = make_vecenv(
        curriculum,
        "serial",
        num_envs=2,
        batch_size=2,
        num_workers=1,
    )

    # Create agent
    agent = create_agent(
        "simple_cnn",
        vecenv.single_observation_space,
        vecenv.single_action_space,
        hidden_size=32,
        device=device,
    )

    # Create experience buffer
    experience = Experience(
        total_agents=vecenv.num_agents,
        batch_size=128,
        bptt_horizon=8,
        minibatch_size=32,
        obs_space=vecenv.single_observation_space,
        atn_space=vecenv.single_action_space,
        device=device,
        hidden_size=32,
    )

    # Create components
    collector = RolloutCollector(vecenv, agent, experience, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    ppo = PPOOptimizer(agent, optimizer, device, PPOConfig())

    # Run a few iterations
    for i in range(3):
        stats, steps = collector.collect()
        print(f"  Iteration {i + 1}: collected {steps} steps")

        losses = ppo.update(experience, update_epochs=1)
        print(f"  Policy loss: {losses['policy_loss']:.4f}")

    vecenv.close()
    print("✓ Direct component usage successful!")

    return True


if __name__ == "__main__":
    print("=== Metta Functional Training Tests ===\n")

    try:
        # Run tests
        test_basic_training()
        test_custom_loss()
        test_direct_components()

        print("\n✅ All tests passed! The functional training system is working correctly.")
        print("\nYou can now run the full examples:")
        print("  python examples/run_functional_training.py")
        print("  python examples/functional_training_example.py")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
