"""Supervised learning training loop for imitating Nim scripted agent."""

import importlib
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cogames.cli.mission import get_mission
from mettagrid.mettagrid_c import PackedCoordinate
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.stateless import StatelessPolicyNet
from mettagrid.simulator import Simulation

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.join(current_dir, "bindings/generated")
if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)

fa = importlib.import_module("fast_agents")

# ENV CONFIG
MISSION_NAME = "evals.extractor_hub_30"
VARIANT = "lonely_heart"

# TRAINING CONFIG
NUM_AGENTS = 1
BATCH_SIZE = 256
NUM_ENVS = 8
NUM_STEPS = 100000 * BATCH_SIZE
LEARNING_RATE = 1e-4
DEVICE = "cpu"
CHECKPOINT_DIR = Path("./test_train_dir")
EXPERIENCE_PER_GRAD_STEP = BATCH_SIZE * NUM_AGENTS * NUM_ENVS
print(f"Experience per gradient step: {EXPERIENCE_PER_GRAD_STEP}")
LOG_INTERVAL = EXPERIENCE_PER_GRAD_STEP


def convert_raw_obs_to_policy_tokens(raw_obs: np.ndarray, expected_num_tokens: int) -> np.ndarray:
    """Convert simulator raw observations into the token layout used by StatelessPolicyNet.

    Input format (raw_obs from simulator):
        Shape: (num_tokens, 3) where each row is [packed_coord, feature_id, value]
        - packed_coord: Packed coordinate byte from PackedCoordinate encoding (row, col) coordinates.
          Can be unpacked using PackedCoordinate.unpack() to get (row, col) tuple.
        - feature_id: Feature/attribute identifier (0xFF indicates end of sequence, stops processing)
        - value: Feature value
        - Variable length: Sequence terminates when feature_id == 0xFF

    Output format (policy tokens for StatelessPolicyNet):
        Shape: (expected_num_tokens, 3) where each row is [coords_byte, feature_id, value]
        - coords_byte: Single byte encoding coordinates with col in high nibble (bits 4-7) and
          row in low nibble (bits 0-3). Formula: ((col & 0x0F) << 4) | (row & 0x0F)
        - feature_id: Same as input (feature/attribute identifier)
        - value: Same as input (feature value)
        - Fixed length: Padded to expected_num_tokens with [255, 0, 0] padding tokens
          (coords_byte=255 indicates padding/invalid token)

    The conversion unpacks the PackedCoordinate format and re-encodes coordinates into the
    format expected by StatelessPolicyNet (matching StatelessAgentPolicyImpl.step()), while
    preserving feature_id and value.
    """

    tokens = []
    for packed_coord, feature_id, value in raw_obs:
        if feature_id == 0xFF:
            break

        location = PackedCoordinate.unpack(int(packed_coord)) or (0, 0)
        col, row = location
        coords_byte = ((col & 0x0F) << 4) | (row & 0x0F)
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
    num_envs: int = NUM_ENVS,
    learning_rate: float = LEARNING_RATE,
    device: str = DEVICE,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    train_from_scratch: bool = False,
):
    """Train a policy to imitate the Nim scripted agent using supervised learning.

    Args:
        mission_name: Name of the mission to train on
        variant: Mission variant (e.g., "lonely_heart")
        num_agents: Number of agents in the environment
        num_steps: Number of training steps
        batch_size: Number of parallel environments to collect observations from
        num_envs: Number of environments to train on
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
    """
    torch_device = torch.device(device)

    # Initialize environment
    variants_arg = [variant] if variant else None
    _, env_cfg, _ = get_mission(mission_name, variants_arg=variants_arg, cogs=num_agents)

    # Create PolicyEnvInterface from config
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

    # Initialize model using StatelessPolicyNet for compatibility with cogames play
    obs_shape = policy_env_info.observation_space.shape
    model = StatelessPolicyNet(policy_env_info.actions, obs_shape)
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # load model weights if available
    if not train_from_scratch:
        checkpoint_path = checkpoint_dir / "policy.pt"
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch_device))
            model = model.to(torch_device)

    # Create multiple simulations for batching
    simulations = []  # list of environments
    teacher_agent_policies_list = []  # list of nim teacher policies per environment as these have state

    for _ in range(NUM_ENVS):
        sim = Simulation(env_cfg, seed=random.randint(0, 2**31 - 1))
        simulations.append(sim)
        policies = [fa.ThinkyAgent(agent_id, policy_env_info.to_json()) for agent_id in range(num_agents)]
        teacher_agent_policies_list.append(policies)

    """
    We now should have a nested list of teacher agent policies for each environment.
    [
        [policy_agent_1_env1, policy_agent_2_env1, policy_agent_3_env1],
        [policy_agent_1_env2, policy_agent_2_env2, policy_agent_3_env2],
        [policy_agent_1_env3, policy_agent_2_env3, policy_agent_3_env3],
        [policy_agent_1_env4, policy_agent_2_env4, policy_agent_3_env4],
        [policy_agent_1_env5, policy_agent_2_env5, policy_agent_3_env5],
        [policy_agent_1_env6, policy_agent_2_env6, policy_agent_3_env6],
        [policy_agent_1_env7, policy_agent_2_env7, policy_agent_3_env7],
        [policy_agent_1_env8, policy_agent_2_env8, policy_agent_3_env8],
        ...
    ]
    """
    assert len(teacher_agent_policies_list) == NUM_ENVS
    assert all(len(policies) == num_agents for policies in teacher_agent_policies_list)

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

    #########################################################
    # Collect (obs, action) pairs
    #########################################################
    while step_count < num_steps:
        # Collect observations and actions from all environments
        obs_batch: list[np.ndarray] = []
        action_batch: list[int] = []

        for _ in range(batch_size):
            # Iterate over all environments
            for env_idx, _ in enumerate(simulations):
                # Reset both the simulation and the nim policy if episode has ended
                if simulations[env_idx].is_done():
                    # Reset simulation (no reset() method, so close and recreate)
                    simulations[env_idx].close()
                    simulations[env_idx] = Simulation(env_cfg, seed=random.randint(0, 2**31 - 1))

                    # Reset nim policy (might not be necessary)
                    for policy in teacher_agent_policies_list[env_idx]:
                        policy.reset()

                # Get the current simulation
                sim = simulations[env_idx]

                # Get observations from environment (these tokens are now in the C++ y high packing)
                sim_raw_observations = sim._c_sim.observations()  # Shape: (num_agents, num_tokens, 3)

                # "forward pass" of the nim policy for all agents
                action_array = sim._c_sim.actions()
                for agent_idx in range(num_agents):
                    # Get the current agent policy
                    agent_policy = teacher_agent_policies_list[env_idx][agent_idx]

                    # Get the observation for the current agent
                    raw_obs = sim_raw_observations[agent_idx]  # Shape: (num_tokens, 3)

                    # Get the action for the current agent
                    agent_policy.step(
                        num_agents=num_agents,
                        num_tokens=200,
                        size_token=3,
                        raw_observations=raw_obs.ctypes.data,
                        num_actions=num_agents,
                        raw_actions=action_array.ctypes.data,
                    )

                    # Get observation in shape expected by StatelessPolicyNet: (num_tokens, token_dim)
                    obs_tokens = convert_raw_obs_to_policy_tokens(raw_obs, 200)

                    obs_batch.append(obs_tokens)
                    action_batch.append(action_array[agent_idx])

                # step environment
                sim.step()

        """
        At this point we should have a two lists:
        - obs_batch: list of (num_tokens, token_dim) tensors
        - action_batch: list of action ids

        Both of these lists should have length batch_size * num_agents * num_envs.
        """
        assert len(obs_batch) == batch_size * num_agents * num_envs
        assert len(action_batch) == batch_size * num_agents * num_envs

        #########################################################
        # Train student
        #########################################################
        # Convert to batched tensors
        # Shape: (batch_size * num_agents * num_envs, num_tokens, token_dim)
        obs_batch_tensor = torch.stack([torch.from_numpy(obs).float() for obs in obs_batch]).to(torch_device)
        # Shape: (batch_size * num_agents * num_envs,)
        action_batch_tensor = torch.tensor(action_batch, dtype=torch.long).to(torch_device)

        # Forward pass through student model
        logits, _ = model.forward_eval(obs_batch_tensor)  # Shape: (batch_size * num_agents * num_envs, num_actions)

        assert logits.shape == (batch_size * num_agents * num_envs, len(policy_env_info.actions.actions()))

        # Compute loss
        loss = loss_fn(logits, action_batch_tensor)
        predictions = torch.argmax(logits, dim=1)
        log_interval_correct += int((predictions == action_batch_tensor).sum().item())
        log_interval_total += len(action_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_count += len(action_batch)

        # Logging
        if step_count % LOG_INTERVAL == 0:
            current_time = time.perf_counter()
            elapsed_since_last_log = current_time - last_log_time
            steps_since_last_log = step_count - last_log_steps

            if elapsed_since_last_log > 0:
                steps_per_second = steps_since_last_log / elapsed_since_last_log
            else:
                steps_per_second = 0.0

            accuracy = (log_interval_correct / log_interval_total) if log_interval_total else 0.0
            print(
                f"Step {step_count}/{num_steps}, Loss: {loss.item():.4f}, "
                f"Accuracy: {accuracy:.3f}, Steps/s: {steps_per_second:.1f}"
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
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="Number of environments to train on")
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        default=False,
        help="Train a new model from scratch without loading checkpoint",
    )

    args = parser.parse_args()

    train_supervised(
        mission_name=args.mission,
        variant=args.variant,
        num_agents=args.agents,
        num_steps=args.steps,
        batch_size=args.batch_size,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
        train_from_scratch=args.train_from_scratch,
    )
