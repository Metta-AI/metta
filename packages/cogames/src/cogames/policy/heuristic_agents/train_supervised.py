"""Supervised learning training loop for imitating Nim scripted agent."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from cogames.cli.mission import get_mission
from cogames.policy.heuristic_agents.simple_nim_agents import HeuristicAgentsPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulation

# CONSTANTS
INPUT_SIZE = 200 * 3
HIDDEN_SIZE = 256
OUTPUT_SIZE = 12

MISSION_NAME = "evals.extractor_hub_30"
VARIANT = "lonely_heart"
NUM_AGENTS = 1
NUM_STEPS = 10000
LEARNING_RATE = 1e-4
DEVICE = "cpu"
CHECKPOINT_DIR = Path("./train_dir")
SEED = 42
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 1000


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Return logits, not softmax (CrossEntropyLoss includes softmax)


def train_supervised(
    mission_name: str = MISSION_NAME,
    variant: Optional[str] = VARIANT,
    num_agents: int = NUM_AGENTS,
    num_steps: int = NUM_STEPS,
    learning_rate: float = LEARNING_RATE,
    device: str = DEVICE,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    seed: int = SEED,
):
    """Train a policy to imitate the Nim scripted agent using supervised learning.

    Args:
        mission_name: Name of the mission to train on
        variant: Mission variant (e.g., "lonely_heart")
        num_agents: Number of agents in the environment
        num_steps: Number of training steps
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
        seed: Random seed
    """
    device = torch.device(device)

    # Initialize environment
    variants_arg = [variant] if variant else None
    _, env_cfg, _ = get_mission(mission_name, variants_arg=variants_arg, cogs=num_agents)

    # Create PolicyEnvInterface from config
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

    # Create scripted agent policy (teacher)
    scripted_policy = HeuristicAgentsPolicy(policy_env_info)

    # Initialize model
    model = SimpleMLP()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Create simulation for environment interaction
    sim = Simulation(env_cfg, seed=seed)

    # Create scripted agent policies for getting teacher actions
    scripted_agent_policies = [scripted_policy.agent_policy(agent_id) for agent_id in range(num_agents)]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step_count = 0
    episode_count = 0

    print(f"Starting supervised training on {mission_name}")
    if variant:
        print(f"Variant: {variant}")
    print(f"Device: {device}, Agents: {num_agents}")

    while step_count < num_steps:
        # Check if episode is done, reset if needed
        if sim.is_done():
            episode_count += 1
            sim.close()
            sim = Simulation(env_cfg, seed=seed + episode_count)
            # Reset agent policies
            for policy in scripted_agent_policies:
                policy.reset(simulation=sim)
            continue

        # Get observations from environment
        raw_obs = sim.raw_observations()  # Shape: (num_agents, num_tokens, 3)
        raw_action = sim.raw_actions()  # Shape: (num_agents,)

        # Get scripted agent's action (teacher)
        # The scripted agent writes its action to raw_action array
        scripted_agent_policies[0].step(raw_obs, raw_action)
        teacher_action = int(raw_action[0])  # Get action for agent 0

        # Convert observation to tensor for model
        # raw_obs[0] is shape (num_tokens, 3) - flatten to (num_tokens * 3,)
        obs_flat = raw_obs[0].flatten()  # Flatten to 1D array
        if len(obs_flat) != INPUT_SIZE:
            raise ValueError(f"Observation length {len(obs_flat)} is not equal to INPUT_SIZE {INPUT_SIZE}")
        obs_tensor = torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)  # Add batch dimension

        # Forward pass through model
        logits = model(obs_tensor)  # Shape: (1, num_actions)

        # Compute loss
        teacher_action_tensor = torch.tensor(teacher_action, dtype=torch.long).to(device)
        loss = loss_fn(logits, teacher_action_tensor.unsqueeze(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step environment with scripted agent's action (already set in raw_action)
        sim.step()

        step_count += 1

        # Logging
        if step_count % LOG_INTERVAL == 0:
            print(f"Step {step_count}/{num_steps}, Loss: {loss.item():.4f}, Episode: {episode_count}")

        # Save checkpoint periodically
        if step_count % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = checkpoint_dir / f"supervised_{step_count}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final checkpoint
    final_checkpoint = checkpoint_dir / "supervised_final.pt"
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Training complete! Final checkpoint saved to {final_checkpoint}")

    sim.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Supervised training to imitate Nim scripted agent")
    parser.add_argument("--mission", type=str, default=MISSION_NAME, help="Mission name")
    parser.add_argument("--variant", type=str, default=VARIANT, help="Mission variant")
    parser.add_argument("--agents", type=int, default=NUM_AGENTS, help="Number of agents")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cpu/cuda)")
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR), help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")

    args = parser.parse_args()

    train_supervised(
        mission_name=args.mission,
        variant=args.variant,
        num_agents=args.agents,
        num_steps=args.steps,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
        seed=args.seed,
    )
