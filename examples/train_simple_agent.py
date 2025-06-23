#!/usr/bin/env python3
"""Example script showing how to train agents without Hydra configuration."""

import torch

from metta.agent import LargeCNNAgent, SimpleCNNAgent, create_agent
from metta.train.config import AgentConfig, TrainerConfig, TrainingConfig
from metta.train.simple_trainer import train_agent
from mettagrid import MettaGridEnv


def train_simple_cnn_example():
    """Example of training a simple CNN agent."""
    print("Training Simple CNN Agent...")

    # Option 1: Using the convenience function
    agent = train_agent(
        agent_name="simple_cnn",
        total_timesteps=1_000_000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        wandb_enabled=False,  # Disable wandb for this example
    )

    print(f"Training complete! Agent has {agent.total_params:,} parameters")
    return agent


def train_custom_agent_example():
    """Example of training with custom configuration."""
    print("Training Custom Large CNN Agent...")

    # Create custom configuration
    config = TrainingConfig(
        run_name="custom_large_cnn_run",
        agent=AgentConfig(
            name="large_cnn",
            hidden_size=512,
            lstm_layers=3,
            kwargs={"num_attention_heads": 16},  # Agent-specific params
        ),
        trainer=TrainerConfig(
            total_timesteps=5_000_000,
            batch_size=65536,
            learning_rate=1e-4,
            checkpoint_interval=30,
        ),
    )

    # Create environment with custom settings
    env = MettaGridEnv(
        width=15,
        height=15,
        max_steps=1000,
    )

    # Create and train agent
    from metta.train.simple_trainer import SimpleTrainer

    trainer = SimpleTrainer(config, env)
    trainer.train()

    return trainer.policy


def create_agent_programmatically():
    """Example of creating agents programmatically without training."""
    print("Creating agents programmatically...")

    # Create a simple environment to get the specs
    env = MettaGridEnv()

    # Create different agent architectures
    agents = {}

    # Simple CNN agent
    agents["simple"] = SimpleCNNAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        device="cpu",
    )

    # Large CNN agent with custom hidden size
    agents["large"] = LargeCNNAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        device="cpu",
        hidden_size=1024,
        lstm_layers=4,
    )

    # Using the factory function
    agents["attention"] = create_agent(
        agent_name="attention",
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        device="cpu",
        num_attention_heads=4,
    )

    # Print agent information
    for name, agent in agents.items():
        print(f"\n{name.capitalize()} Agent:")
        print(f"  Total parameters: {agent.total_params:,}")
        print(f"  Hidden size: {agent.hidden_size}")
        print(f"  LSTM layers: {agent.lstm_layers}")

    return agents


def custom_agent_architecture():
    """Example of creating a custom agent architecture."""
    import torch.nn as nn

    from metta.agent import BaseAgent, register_agent
    from metta.agent.components.observation_normalizer import ObservationNormalizer

    class CustomAgent(BaseAgent):
        """Custom agent with residual connections."""

        def __init__(
            self, obs_space, action_space, obs_width, obs_height, feature_normalizations, device="cuda", **kwargs
        ):
            super().__init__(obs_space, action_space, device=device)

            # Custom architecture with residual connections
            obs_shape = obs_space.shape if hasattr(obs_space, "shape") else obs_space.spaces["grid_obs"].shape

            self.obs_normalizer = ObservationNormalizer(
                input_shape=obs_shape, feature_normalizations=feature_normalizations
            )

            # Residual CNN blocks
            self.conv1 = nn.Conv2d(obs_shape[-1], 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

            # ... rest of the architecture

        def compute_outputs(self, x, state):
            # Custom forward pass
            # ... implementation
            pass

    # Register the custom agent
    register_agent("custom_residual", CustomAgent)

    # Now it can be created using the factory
    env = MettaGridEnv()
    agent = create_agent(
        "custom_residual",
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
    )

    return agent


if __name__ == "__main__":
    # Example 1: Train a simple CNN agent
    simple_agent = train_simple_cnn_example()

    # Example 2: Create agents without training
    agents = create_agent_programmatically()

    # Example 3: Train with custom configuration
    # custom_agent = train_custom_agent_example()  # Uncomment to run

    print("\nAll examples completed!")
