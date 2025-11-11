"""Supervised learning training loop for imitating Nim scripted agent."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cogames.cli.mission import get_mission
from cogames.policy.heuristic_agents.simple_nim_agents import HeuristicAgentsPolicy
from mettagrid.mettagrid_c import PackedCoordinate
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.stateless import StatelessPolicyNet
from mettagrid.simulator import Simulation

# CONSTANTS
MISSION_NAME = "evals.extractor_hub_30"
VARIANT = "lonely_heart"
NUM_AGENTS = 1
BATCH_SIZE = 256
NUM_STEPS = 10000 * BATCH_SIZE
LEARNING_RATE = 1e-4
DEVICE = "cpu"
CHECKPOINT_DIR = Path("./test_train_dir")
SEED = 42
LOG_INTERVAL = 1000
NIM_DEBUG = False


def convert_raw_obs_to_policy_tokens(raw_obs: np.ndarray, expected_num_tokens: int) -> np.ndarray:
    """Convert simulator raw observations into the token layout used by StatelessPolicyNet."""

    tokens = []
    for packed_coord, feature_id, value in raw_obs:
        if feature_id == 0xFF:
            break

        location = PackedCoordinate.unpack(int(packed_coord)) or (0, 0)
        row, col = location
        # StatelessPolicyImpl encodes coords with row in the high nibble and col in the low nibble
        coords_byte = ((row & 0x0F) << 4) | (col & 0x0F)
        tokens.append([coords_byte, int(feature_id), int(value)])

    if len(tokens) < expected_num_tokens:
        tokens.extend([[255, 0, 0]] * (expected_num_tokens - len(tokens)))

    return np.array(tokens, dtype=np.uint8)


def train_supervised(
    mission_name: str = MISSION_NAME,
    variant: Optional[str] = VARIANT,
    num_agents: int = NUM_AGENTS,
    num_steps: int = NUM_STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    device: str = DEVICE,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    seed: int = SEED,
    nim_debug: bool = NIM_DEBUG,
):
    """Train a policy to imitate the Nim scripted agent using supervised learning.

    Args:
        mission_name: Name of the mission to train on
        variant: Mission variant (e.g., "lonely_heart")
        num_agents: Number of agents in the environment
        num_steps: Number of training steps
        batch_size: Number of parallel environments to collect observations from
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
        seed: Random seed
        nim_debug: Enable debug prints from Nim heuristic agent
    """
    torch_device = torch.device(device)

    # Initialize environment
    variants_arg = [variant] if variant else None
    _, env_cfg, _ = get_mission(mission_name, variants_arg=variants_arg, cogs=num_agents)

    # Create PolicyEnvInterface from config
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

    # Create scripted agent policy (teacher)
    scripted_policy = HeuristicAgentsPolicy(policy_env_info, debug=nim_debug)

    # Initialize model using StatelessPolicyNet for compatibility with cogames play
    obs_shape = policy_env_info.observation_space.shape
    expected_num_tokens = obs_shape[0]
    model = StatelessPolicyNet(policy_env_info.actions, obs_shape)
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # load model weights if available
    checkpoint_path = checkpoint_dir / "policy.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch_device))
        model = model.to(torch_device)

    # Create multiple simulations for batching
    simulations = []
    scripted_agent_policies_list = []
    episode_counts = []

    for i in range(batch_size):
        sim = Simulation(env_cfg, seed=seed + i)
        simulations.append(sim)
        policies = [scripted_policy.agent_policy(agent_id) for agent_id in range(num_agents)]
        scripted_agent_policies_list.append(policies)
        episode_counts.append(0)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step_count = 0
    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_steps = 0
    log_interval_correct = 0
    log_interval_total = 0

    print(f"Starting supervised training on {mission_name}")
    if variant:
        print(f"Variant: {variant}")
    print(f"Device: {device}, Agents: {num_agents}, Batch size: {batch_size}")

    while step_count < num_steps:
        # Collect observations and actions from all environments
        obs_batch = []
        action_batch = []
        valid_indices = []

        for env_idx, sim in enumerate(simulations):
            # Reset if episode is done
            if sim.is_done():
                episode_counts[env_idx] += 1
                sim.close()
                simulations[env_idx] = Simulation(env_cfg, seed=seed + env_idx + episode_counts[env_idx] * batch_size)
                for policy in scripted_agent_policies_list[env_idx]:
                    policy.reset(simulation=simulations[env_idx])
                sim = simulations[env_idx]

            # Get observations from environment
            raw_obs = sim.raw_observations()  # Shape: (num_agents, num_tokens, 3)
            raw_action = sim.raw_actions()  # Shape: (num_agents,)

            # Get scripted agent's action (teacher)
            scripted_agent_policies_list[env_idx][0].step(raw_obs, raw_action)
            teacher_action = int(raw_action[0])

            # Get observation in shape expected by StatelessPolicyNet: (num_tokens, token_dim)
            obs_tokens = convert_raw_obs_to_policy_tokens(raw_obs[0], expected_num_tokens)

            obs_batch.append(obs_tokens)
            action_batch.append(teacher_action)
            valid_indices.append(env_idx)

        # Skip if no valid environments (all done)
        if not valid_indices:
            continue

        # Convert to batched tensors
        # Shape: (batch_size, num_tokens, token_dim)
        obs_batch_tensor = torch.stack([torch.from_numpy(obs).float() for obs in obs_batch]).to(torch_device)
        # Shape: (batch_size,)
        action_batch_tensor = torch.tensor(action_batch, dtype=torch.long).to(torch_device)

        # Forward pass through model (StatelessPolicyNet divides by 255.0 internally)
        logits, _ = model.forward_eval(obs_batch_tensor)  # Shape: (batch_size, num_actions)

        # Compute loss
        loss = loss_fn(logits, action_batch_tensor)
        predictions = torch.argmax(logits, dim=1)
        log_interval_correct += int((predictions == action_batch_tensor).sum().item())
        log_interval_total += len(valid_indices)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step all environments
        for env_idx in valid_indices:
            simulations[env_idx].step()

        step_count += len(valid_indices)

        # Logging
        if step_count % LOG_INTERVAL == 0:
            current_time = time.perf_counter()
            elapsed_since_last_log = current_time - last_log_time
            steps_since_last_log = step_count - last_log_steps

            if elapsed_since_last_log > 0:
                steps_per_second = steps_since_last_log / elapsed_since_last_log
            else:
                steps_per_second = 0.0

            avg_episode = sum(episode_counts) / len(episode_counts) if episode_counts else 0
            accuracy = (log_interval_correct / log_interval_total) if log_interval_total else 0.0
            print(
                f"Step {step_count}/{num_steps}, Loss: {loss.item():.4f}, "
                f"Accuracy: {accuracy:.3f}, Avg Episode: {avg_episode:.1f}, Steps/s: {steps_per_second:.1f}"
            )

            last_log_time = current_time
            last_log_steps = step_count
            log_interval_correct = 0
            log_interval_total = 0

    # Final checkpoint - save as policy.pt for compatibility with cogames play
    final_checkpoint = checkpoint_dir / "policy.pt"
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Training complete! Final checkpoint saved to {final_checkpoint}")
    print(f"Load with: uv run cogames play -m {mission_name} -p stateless:{final_checkpoint}")

    # Close all simulations
    for sim in simulations:
        sim.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Supervised training to imitate Nim scripted agent")
    parser.add_argument("--mission", type=str, default=MISSION_NAME, help="Mission name")
    parser.add_argument("--variant", type=str, default=VARIANT, help="Mission variant")
    parser.add_argument("--agents", type=int, default=NUM_AGENTS, help="Number of agents")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of training steps")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size (number of parallel environments)"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cpu/cuda)")
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR), help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--nim-debug", action="store_true", default=NIM_DEBUG, help="Enable debug prints from Nim heuristic agent"
    )

    args = parser.parse_args()

    train_supervised(
        mission_name=args.mission,
        variant=args.variant,
        num_agents=args.agents,
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
        seed=args.seed,
        nim_debug=args.nim_debug,
    )
