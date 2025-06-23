#!/usr/bin/env python3
"""Examples of using Metta as a library without YAML configuration.

This demonstrates the refactored approach where components are used
directly without Hydra or complex configuration files.
"""

from pathlib import Path

import torch


# Example 1: Simplest possible training
def example_minimal():
    """Minimal example - just train an agent."""
    from metta.train.job import quick_train

    # Train with defaults
    agent = quick_train("simple_cnn", total_timesteps=100_000)
    print(f"Trained agent with {agent.total_params:,} parameters")


# Example 2: Custom agent and environment
def example_custom_agent_env():
    """Create custom agent and environment without configs."""
    from metta.agent import SimpleCNNAgent
    from metta.env.factory import create_env_from_preset
    from metta.runtime import configure

    # Configure runtime (replaces common.yaml)
    runtime = configure(
        run_name="my_custom_run",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )

    # Create environment
    env = create_env_from_preset("medium", num_agents=2)

    # Create agent directly
    agent = SimpleCNNAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        device=runtime.device,
        hidden_size=256,  # Custom size
    )

    # Activate actions
    agent.activate_actions(env.action_names, env.max_action_args, runtime.device)

    print(f"Created {agent.__class__.__name__} for {env.__class__.__name__}")
    return agent, env


# Example 3: Fluent training job builder
def example_job_builder():
    """Use the fluent builder interface for training."""
    from metta.train.job import JobBuilder

    # Build a complete training job
    agent = (
        JobBuilder()
        .with_agent("large_cnn")
        .with_timesteps(5_000_000)
        .with_batch_size(65536)
        .with_evaluations("navigation")  # Just navigation tasks
        .with_wandb("my_project", entity="my_team")
        .with_device("cuda:0")
        .run()
    )

    return agent


# Example 4: Direct component composition
def example_direct_composition():
    """Compose training components directly."""
    from metta.agent import create_agent
    from metta.env.factory import create_vectorized_env
    from metta.runtime import RuntimeConfig
    from metta.train.config import AgentConfig, TrainerConfig, TrainingConfig

    # Runtime configuration
    runtime = RuntimeConfig(
        run_name="direct_composition_example",
        data_dir=Path("./experiments"),
    )

    # Training configuration
    config = TrainingConfig(
        runtime=runtime,
        trainer=TrainerConfig(
            total_timesteps=10_000_000,
            batch_size=32768,
            learning_rate=3e-4,
            checkpoint_interval=30,
        ),
        agent=AgentConfig(
            name="attention",
            hidden_size=512,
            kwargs={"num_attention_heads": 16},
        ),
    )

    # Create vectorized environment
    vecenv = create_vectorized_env(
        num_envs=128,
        num_workers=8,
        device=str(runtime.device),
        width=15,
        height=15,
    )

    # Create agent
    env = vecenv.driver_env
    agent = create_agent(
        config.agent.name,
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        device=str(runtime.device),
        **config.agent.kwargs,
    )

    print(f"Created {config.agent.name} agent with vectorized env")
    return agent, vecenv


# Example 5: Custom simulation suites
def example_custom_simulations():
    """Create custom evaluation suites programmatically."""
    from metta.sim.registry import SimulationRegistry, SimulationSpec
    from metta.train.job import TrainingJob

    # Create custom registry
    registry = SimulationRegistry()

    # Register custom simulations
    registry.register(
        SimulationSpec(
            name="custom/easy_nav",
            env="env/mettagrid/navigation/evals/emptyspace_withinsight",
            num_episodes=10,
            max_time_s=60,
        )
    )

    registry.register(
        SimulationSpec(
            name="custom/hard_nav",
            env="env/mettagrid/navigation/evals/labyrinth",
            num_episodes=5,
            max_time_s=180,
        )
    )

    # Create a custom suite
    registry.register_suite("my_suite", ["custom/easy_nav", "custom/hard_nav"])

    # Use in training
    job = TrainingJob(
        agent="simple_cnn",
        evaluations=registry.get_suite("my_suite"),
    )

    print(f"Created custom evaluation suite with {len(job.evaluations.simulations)} simulations")


# Example 6: Standalone evaluation
def example_standalone_eval():
    """Run evaluation without training."""
    from metta.agent import create_agent
    from metta.env.factory import create_env
    from metta.sim.simulation import Simulation
    from metta.sim.simulation_config import SingleEnvSimulationConfig

    # Create environment and agent
    env = create_env()
    agent = create_agent(
        "simple_cnn",
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        device="cpu",
    )

    # Create simulation config
    sim_config = SingleEnvSimulationConfig(
        env="env/mettagrid/simple",
        num_episodes=10,
        max_time_s=60,
    )

    # Run simulation
    sim = Simulation(
        sim_name="test_eval",
        sim_config=sim_config,
        policy_agent=agent,
        device="cpu",
    )

    # results = sim.run()  # Would run the simulation
    print("Created standalone simulation")


# Example 7: Training with custom loop
def example_custom_training_loop():
    """Implement custom training loop for maximum control."""
    from metta.agent import SimpleCNNAgent
    from metta.env.factory import create_vectorized_env
    from metta.runtime import configure

    # Setup
    runtime = configure(run_name="custom_loop")

    # Create components
    vecenv = create_vectorized_env(num_envs=32, width=11, height=11)

    agent = SimpleCNNAgent(
        obs_space=vecenv.driver_env.observation_space,
        action_space=vecenv.driver_env.action_space,
        obs_width=11,
        obs_height=11,
        feature_normalizations={},
        device=runtime.device,
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    # Pseudo-code for custom loop
    """
    rollout_manager = RolloutManager(vecenv, agent)
    ppo_update = PPOUpdate(clip_param=0.2, value_coef=0.5, entropy_coef=0.01)

    for epoch in range(1000):
        # Collect rollouts
        rollouts = rollout_manager.collect(steps=2048)

        # Compute advantages
        advantages = compute_advantages(rollouts)

        # PPO update
        losses = ppo_update(agent, rollouts, advantages, optimizer)

        # Log and checkpoint
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {losses}")
    """

    print("Custom training loop components created")


# Example 8: Integration with existing code
def example_integration():
    """Show how to integrate with existing PyTorch code."""
    import torch.nn as nn

    from metta.agent import BaseAgent, register_agent

    # Existing PyTorch model
    class MyExistingModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    # Wrap as Metta agent
    class MyExistingAgent(BaseAgent):
        def __init__(self, obs_space, action_space, **kwargs):
            super().__init__(obs_space, action_space, **kwargs)

            # Use existing model
            self.encoder = MyExistingModel(
                input_dim=obs_space.shape[0] * obs_space.shape[1] * obs_space.shape[2],
                hidden_dim=256,
                output_dim=self.hidden_size,
            )

            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.lstm_layers)
            self.value_head = nn.Linear(self.hidden_size, 1)
            self.actor_head = nn.Linear(self.hidden_size, 100)  # Placeholder

        def compute_outputs(self, x, state):
            # Flatten and encode
            x = x.flatten(1)
            x = self.encoder(x)

            # LSTM
            x, (h, c) = self.lstm(x.unsqueeze(0), (state.lstm_h, state.lstm_c))
            x = x.squeeze(0)

            # Heads
            value = self.value_head(x)
            logits = self.actor_head(x)

            return value, logits, (h, c)

    # Register and use
    register_agent("my_existing", MyExistingAgent)

    # agent = quick_train("my_existing", total_timesteps=100_000)

    print("Integrated existing PyTorch model as Metta agent")


if __name__ == "__main__":
    print("=== Metta Direct Library Usage Examples ===\n")

    print("1. Minimal training example:")
    example_minimal()

    print("\n2. Custom agent and environment:")
    example_custom_agent_env()

    print("\n3. Fluent job builder:")
    # example_job_builder()  # Commented to avoid long training

    print("\n4. Direct component composition:")
    example_direct_composition()

    print("\n5. Custom simulation suites:")
    example_custom_simulations()

    print("\n6. Standalone evaluation:")
    example_standalone_eval()

    print("\n7. Custom training loop:")
    example_custom_training_loop()

    print("\n8. Integration with existing code:")
    example_integration()

    print("\n✓ All examples demonstrate YAML-free usage!")
