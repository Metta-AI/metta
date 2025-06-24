#!/usr/bin/env python3
"""Complete example of using Metta as a library.

This demonstrates all the major features of the refactored library API,
showing how components work together without any YAML configuration.
"""

from pathlib import Path

import torch
import torch.nn as nn

# All imports from the main metta package
import metta


def basic_usage():
    """The simplest way to use Metta."""
    print("=== Basic Usage ===\n")

    # One-line training
    agent = metta.train(metta.SimpleCNNAgent(obs_width=11, obs_height=11, feature_normalizations={}), timesteps=100_000)

    print(f"Trained a {agent.__class__.__name__} with {agent.total_params:,} parameters\n")


def intermediate_usage():
    """More control over the training process."""
    print("=== Intermediate Usage ===\n")

    # Configure runtime environment
    runtime = metta.configure(
        run_name="intermediate_example",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        data_dir=Path("./experiments"),
    )
    print(f"Runtime configured: {runtime.run_name} on {runtime.device}")

    # Create environment from preset
    env = metta.create_env_from_preset("medium", num_agents=2)
    print(f"Created environment: {env.__class__.__name__} ({env.width}x{env.height})")

    # Create agent with custom parameters
    agent = metta.create_agent(
        "large_cnn",
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        device=str(runtime.device),
        hidden_size=512,
        lstm_layers=3,
    )
    print(f"Created {agent.__class__.__name__} agent")

    # Use the Metta trainer
    trainer = metta.Metta(
        agent=agent,
        env=env,
        total_timesteps=1_000_000,
        batch_size=32768,
        run_name=runtime.run_name,
    )

    print("\nTraining would start here...")
    # trainer.train()  # Uncomment to actually train


def advanced_usage():
    """Advanced features including custom agents and evaluations."""
    print("\n=== Advanced Usage ===\n")

    # 1. Register a custom agent
    class ResidualCNNAgent(metta.BaseAgent):
        """Custom agent with residual connections."""

        def __init__(self, obs_space, action_space, **kwargs):
            super().__init__(obs_space, action_space, **kwargs)

            # Custom architecture
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)

            self.lstm = nn.LSTM(64, self.hidden_size, self.lstm_layers)
            self.value_head = nn.Linear(self.hidden_size, 1)
            self.policy_head = nn.Linear(self.hidden_size, 100)  # Placeholder

        def compute_outputs(self, x, state):
            # Residual block
            identity = self.conv1(x)
            out = torch.relu(identity)
            out = self.conv2(out)
            out = out + identity  # Residual connection

            # Pool and flatten
            out = self.pool(out).flatten(1)

            # LSTM
            out, (h, c) = self.lstm(out.unsqueeze(0), (state.lstm_h, state.lstm_c))
            out = out.squeeze(0)

            # Heads
            value = self.value_head(out)
            logits = self.policy_head(out)

            return value, logits, (h, c)

    # Register the custom agent
    metta.register_agent("residual_cnn", ResidualCNNAgent)
    print("Registered custom ResidualCNNAgent")

    # 2. Create custom evaluation suite
    metta.register_simulation(
        name="custom/test_nav",
        env="env/mettagrid/navigation/evals/emptyspace_withinsight",
        num_episodes=5,
        max_time_s=60,
    )

    metta.register_simulation(
        name="custom/test_memory",
        env="env/mettagrid/memory/evals/easy",
        num_episodes=3,
        max_time_s=120,
    )

    print("Registered custom simulations")

    # 3. Use job builder for complex training setup
    job = (
        metta.JobBuilder()
        .with_agent("residual_cnn")
        .with_timesteps(5_000_000)
        .with_batch_size(65536)
        .with_evaluations(["custom/test_nav", "custom/test_memory"])
        .with_wandb("my_project", entity="my_team")
        .build()
    )

    print(f"\nBuilt training job: {job.agent} for {job.trainer.total_timesteps:,} steps")
    # job.run()  # Uncomment to run


def experiment_usage():
    """Running experiments with multiple configurations."""
    print("\n=== Experiment Usage ===\n")

    # Define experiment
    exp = metta.Experiment("architecture_comparison")

    # Factory functions for different architectures
    def make_small_agent():
        return metta.SimpleCNNAgent(
            obs_width=11,
            obs_height=11,
            feature_normalizations={},
            hidden_size=128,
        )

    def make_large_agent():
        return metta.LargeCNNAgent(
            obs_width=11,
            obs_height=11,
            feature_normalizations={},
            hidden_size=512,
        )

    def make_attention_agent():
        return metta.AttentionAgent(
            obs_width=11,
            obs_height=11,
            feature_normalizations={},
            num_attention_heads=8,
        )

    # Run experiments (commented to avoid long training)
    """
    for agent_fn, name in [
        (make_small_agent, "small"),
        (make_large_agent, "large"),
        (make_attention_agent, "attention"),
    ]:
        print(f"Running experiment with {name} agent...")
        exp.run(
            agent_fn=agent_fn,
            seeds=[0, 1, 2],
            total_timesteps=1_000_000,
        )

    # Get results
    summary = exp.summary()
    print(f"\nResults: {summary['mean']:.2f} ± {summary['std']:.2f}")
    """

    print("Experiment framework ready for multi-seed runs")


def integration_example():
    """Integrating with existing PyTorch code."""
    print("\n=== Integration Example ===\n")

    # Existing PyTorch model (e.g., from another project)
    class PretrainedVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Pretend this is a complex pretrained model
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, 7, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(128, 256, 3),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.output_dim = 256

    # Wrap it as a Metta agent
    class PretrainedAgent(metta.BaseAgent):
        def __init__(self, obs_space, action_space, pretrained_path=None, **kwargs):
            super().__init__(obs_space, action_space, **kwargs)

            # Use pretrained model as encoder
            self.encoder = PretrainedVisionModel()
            if pretrained_path:
                self.encoder.load_state_dict(torch.load(pretrained_path))
                print(f"Loaded pretrained weights from {pretrained_path}")

            # Freeze pretrained layers if desired
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Add RL-specific layers
            self.lstm = nn.LSTM(self.encoder.output_dim, self.hidden_size, self.lstm_layers)
            self.value_head = nn.Linear(self.hidden_size, 1)
            self.policy_head = nn.Linear(self.hidden_size, 100)

        def compute_outputs(self, x, state):
            # Use pretrained features
            with torch.no_grad():  # Don't compute gradients for frozen layers
                features = self.encoder.features(x).flatten(1)

            # RL layers
            out, (h, c) = self.lstm(features.unsqueeze(0), (state.lstm_h, state.lstm_c))
            out = out.squeeze(0)

            value = self.value_head(out)
            logits = self.policy_head(out)

            return value, logits, (h, c)

    # Use it
    agent = PretrainedAgent(
        obs_space=None,  # Would come from env
        action_space=None,
        obs_width=11,
        obs_height=11,
        feature_normalizations={},
        pretrained_path=None,  # Would load real weights
    )

    print("Created agent with pretrained vision model")
    print(f"Trainable parameters: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in agent.parameters() if not p.requires_grad):,}")


def direct_composition():
    """Direct composition of all components."""
    print("\n=== Direct Composition ===\n")

    # Create all components directly
    env = metta.create_env(
        width=15,
        height=15,
        max_steps=1000,
        num_agents=1,
    )

    agent = metta.SimpleCNNAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        feature_normalizations=env.feature_normalizations,
        hidden_size=256,
    )

    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,
    )

    # Compose into trainer
    trainer = metta.Metta(
        agent=agent,
        env=env,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        total_timesteps=10_000_000,
        on_epoch_end=lambda m: print(f"Epoch {m.epoch} complete"),
        on_checkpoint=lambda m: m.save(f"checkpoint_{m.epoch}.pt"),
    )

    print("Created fully customized training setup")
    print(f"Agent: {agent.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Scheduler: {lr_scheduler.__class__.__name__}")


def list_available_components():
    """Show all available components."""
    print("\n=== Available Components ===\n")

    print("Agents:")
    for agent_name in metta.list_agents():
        print(f"  - {agent_name}")

    print("\nEnvironment Presets:")
    for preset_name in metta.ENV_PRESETS:
        preset = metta.ENV_PRESETS[preset_name]
        print(f"  - {preset_name}: {preset['width']}x{preset['height']}")

    print("\nSimulation Suites:")
    registry = metta.sim.registry.get_registry()
    for suite in ["navigation", "objectuse", "memory", "quick"]:
        try:
            suite_config = registry.get_suite(suite)
            print(f"  - {suite}: {len(suite_config.simulations)} simulations")
        except:
            pass


def main():
    """Run all examples."""
    print("=== Metta Library Usage Examples ===\n")
    print("This demonstrates the refactored library API")
    print("No YAML files or Hydra configuration needed!\n")

    # Run examples
    basic_usage()
    intermediate_usage()
    advanced_usage()
    experiment_usage()
    integration_example()
    direct_composition()
    list_available_components()

    print("\n✅ All examples demonstrate direct Python usage!")
    print("The library is now as easy to use as PyTorch or scikit-learn.")


if __name__ == "__main__":
    main()
