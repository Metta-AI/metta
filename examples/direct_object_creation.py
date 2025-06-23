#!/usr/bin/env python3
"""Direct object creation example - exactly as envisioned in the design doc.

This shows the most pythonic way to use Metta, where you create
objects directly without any configuration files.
"""

import torch
import torch.nn as nn

# Direct imports - no configs needed
from metta.agent import SimpleCNNAgent
from metta.env.factory import create_env
from metta.sim.registry import SimulationSpec
from metta.train.minimal import Metta


def main():
    """Create everything directly - no YAML, no Hydra, just Python."""

    # 1. Create the agent directly
    print("Creating agent...")
    agent = SimpleCNNAgent(
        obs_space=None,  # Will be set from env
        action_space=None,  # Will be set from env
        obs_width=11,
        obs_height=11,
        feature_normalizations={},
        device="cuda" if torch.cuda.is_available() else "cpu",
        hidden_size=256,
        lstm_layers=3,
    )

    # Could also create a custom agent inline
    class MyCustomAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(64, 128, 2)
            self.actor = nn.Linear(128, 10)
            self.critic = nn.Linear(128, 1)

        def forward(self, x, state):
            x = self.encoder(x)
            x = x.flatten(1)
            x, state = self.lstm(x, state)
            return self.actor(x), self.critic(x), state

    # 2. Create environment directly
    print("Creating environment...")
    env = create_env(
        width=15,
        height=15,
        max_steps=1000,
        num_agents=1,
        num_resources=3,
        resource_prob=0.02,
    )

    # 3. Create simulations for evaluation
    print("Creating evaluation suite...")
    easy_nav = SimulationSpec(
        name="custom/easy",
        env="env/mettagrid/navigation/evals/emptyspace_withinsight",
        num_episodes=5,
        max_time_s=60,
    )

    hard_nav = SimulationSpec(
        name="custom/hard",
        env="env/mettagrid/navigation/evals/labyrinth",
        num_episodes=3,
        max_time_s=180,
    )

    # 4. Create the Metta training object
    print("Creating Metta trainer...")
    metta = Metta(
        agent=agent,
        env=env,
        total_timesteps=10_000_000,
        batch_size=32768,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        run_name="my_experiment",
    )

    # 5. Custom training loop (as in the design doc)
    print("\nStarting training loop...")

    while metta.training():
        # One training step
        metta.train(timesteps=100_000)  # Train for 100k steps

        # Custom evaluation
        print(f"\nStep {metta.agent_step}:")
        results = metta.eval(num_episodes=5)
        print(f"  Mean reward: {results['mean_reward']:.2f}")

        # Save checkpoint
        if metta.agent_step % 1_000_000 == 0:
            metta.save(f"checkpoint_{metta.agent_step}.pt")
            print(f"  Saved checkpoint at step {metta.agent_step}")

        # Early stopping
        if results["mean_reward"] > 100:
            print("  Early stopping - target reward achieved!")
            break

    print("\nTraining complete!")

    # 6. Alternative: Even simpler one-liner
    print("\nAlternative simple training:")
    from metta.train.minimal import train

    trained_agent = train(
        SimpleCNNAgent(
            obs_width=11,
            obs_height=11,
            feature_normalizations={},
            hidden_size=128,
        ),
        timesteps=1_000_000,
    )

    print("Done!")


def advanced_example():
    """More advanced direct usage with custom components."""

    # Custom optimizer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Create agent
    agent = SimpleCNNAgent(
        obs_width=11,
        obs_height=11,
        feature_normalizations={},
        hidden_size=512,
    )

    # Custom optimizer and scheduler
    optimizer = AdamW(agent.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)

    # Create Metta with custom components
    metta = Metta(
        agent=agent,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        total_timesteps=10_000_000,
        # Callbacks for custom logic
        on_epoch_end=lambda m: print(f"Epoch {m.epoch} complete"),
        on_checkpoint=lambda m: m.save(f"model_epoch_{m.epoch}.pt"),
    )

    # Train
    metta.train()


def experiment_example():
    """Run experiments with multiple seeds."""
    from metta.train.minimal import Experiment

    # Define agent factory
    def make_agent():
        return SimpleCNNAgent(
            obs_width=11,
            obs_height=11,
            feature_normalizations={},
            hidden_size=128,
        )

    # Run experiment
    exp = Experiment("cnn_baseline")
    exp.run(
        agent_fn=make_agent,
        seeds=[0, 1, 2, 3, 4],
        total_timesteps=5_000_000,
        batch_size=32768,
    )

    # Get results
    summary = exp.summary()
    print(f"Results: {summary['mean']:.2f} ± {summary['std']:.2f}")


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to run other examples
    # advanced_example()
    # experiment_example()
