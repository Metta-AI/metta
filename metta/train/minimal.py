"""Minimal training interface - the simplest way to use Metta.

This provides the most direct interface possible, matching the user's
vision of creating objects directly without configuration files.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from metta.agent import BaseAgent
from metta.env.factory import create_env
from metta.runtime import RuntimeConfig


@dataclass
class Metta:
    """The main Metta interface - simple and direct.

    Example:
        agent = SimpleCNNAgent(...)
        metta = Metta(agent=agent)
        metta.train()
    """

    # Core components
    agent: BaseAgent
    env: Optional[Any] = None

    # Training parameters
    total_timesteps: int = 10_000_000
    batch_size: int = 32768
    learning_rate: float = 3e-4

    # Runtime
    device: str = "cuda"
    run_name: str = "metta_run"
    checkpoint_interval: int = 60

    # Optional components
    optimizer: Optional[torch.optim.Optimizer] = None
    lr_scheduler: Optional[Any] = None

    # Callbacks
    on_epoch_end: Optional[Callable] = None
    on_checkpoint: Optional[Callable] = None

    def __post_init__(self):
        """Initialize components."""
        # Create environment if not provided
        if self.env is None:
            self.env = create_env()

        # Move agent to device
        self.agent = self.agent.to(self.device)

        # Activate actions
        if hasattr(self.env, "action_names"):
            self.agent.activate_actions(self.env.action_names, self.env.max_action_args, torch.device(self.device))

        # Create optimizer if not provided
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate)

        # Training state
        self.epoch = 0
        self.agent_step = 0
        self._trainer = None

    def train(self, timesteps: Optional[int] = None) -> None:
        """Run training for specified timesteps."""
        if timesteps:
            self.total_timesteps = timesteps

        # Create trainer if needed
        if self._trainer is None:
            self._create_trainer()

        # Run training
        self._trainer.train()

    def eval(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent."""
        rewards = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Get action from agent
                with torch.no_grad():
                    action = self.agent.act(obs)

                # Step environment
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)

        return {
            "mean_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
        }

    def save(self, path: str) -> None:
        """Save the agent."""
        torch.save(
            {
                "agent_state_dict": self.agent.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "agent_step": self.agent_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load a saved agent."""
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.agent_step = checkpoint.get("agent_step", 0)

    def training(self) -> bool:
        """Check if still training."""
        return self.agent_step < self.total_timesteps

    def _create_trainer(self):
        """Create the internal trainer."""
        # This is a simplified version - full implementation would
        # properly integrate with MettaTrainer
        runtime = RuntimeConfig(
            run_name=self.run_name,
            device=self.device,
        )

        config = {
            "run": self.run_name,
            "device": self.device,
            "trainer": {
                "total_timesteps": self.total_timesteps,
                "batch_size": self.batch_size,
                "checkpoint_interval": self.checkpoint_interval,
                "optimizer": {
                    "learning_rate": self.learning_rate,
                },
            },
        }

        # Simplified trainer creation
        # self._trainer = MettaTrainer(config, ...)


# Even simpler functional interface
def train(
    agent: BaseAgent, timesteps: int = 1_000_000, batch_size: int = 32768, device: str = "cuda", **kwargs
) -> BaseAgent:
    """The simplest possible training interface.

    Example:
        from metta import train
        from metta.agent import SimpleCNNAgent

        agent = SimpleCNNAgent(...)
        trained_agent = train(agent, timesteps=5_000_000)
    """
    metta = Metta(agent=agent, total_timesteps=timesteps, batch_size=batch_size, device=device, **kwargs)

    metta.train()
    return metta.agent


# Quick experiment runner
class Experiment:
    """Run multiple training experiments easily."""

    def __init__(self, name: str):
        self.name = name
        self.results = []

    def run(self, agent_fn: Callable, seeds: list[int] = [0, 1, 2], **train_kwargs) -> None:
        """Run experiment with multiple seeds."""
        for seed in seeds:
            # Set seed
            torch.manual_seed(seed)

            # Create agent
            agent = agent_fn()

            # Train
            metta = Metta(agent=agent, run_name=f"{self.name}_seed{seed}", **train_kwargs)
            metta.train()

            # Evaluate
            results = metta.eval(num_episodes=100)
            results["seed"] = seed
            self.results.append(results)

    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        all_rewards = [r["mean_reward"] for r in self.results]
        return {
            "mean": sum(all_rewards) / len(all_rewards),
            "std": torch.tensor(all_rewards).std().item(),
            "min": min(all_rewards),
            "max": max(all_rewards),
        }
