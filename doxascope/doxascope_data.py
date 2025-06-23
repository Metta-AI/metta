#!/usr/bin/env python3
"""
Doxascope Data Module

This module provides functionality for:
1. Logging LSTM memory vectors and agent positions during simulation
2. Preprocessing the logged data for neural network training

The goal is to train a "doxascope" neural network that can predict agent movement
from their internal memory state.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from metta.agent.policy_state import PolicyState

logger = logging.getLogger(__name__)


class DoxascopeLogger:
    """Logs memory vectors and position data for training doxascope networks."""

    def __init__(self, doxascope_config, simulation_id: str, policy_name: str = "default_policy"):
        self.enabled = doxascope_config.get("enabled", False)
        if not self.enabled:
            return

        # Use the new directory structure with policy_name
        base_dir = Path(doxascope_config.get("output_dir", "./doxascope/data/raw_data/"))
        self.output_dir = base_dir / policy_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_dir / f"doxascope_data_{simulation_id}.json"
        self.data = []
        self.timestep = 0

        logger.info(f"Doxascope logging enabled for policy '{policy_name}', will save raw data to {self.output_file}")

    def log_timestep(
        self,
        policy_state: PolicyState,
        policy_idxs: torch.Tensor,
        env_grid_objects: Dict,
    ):
        """Log memory vectors and positions for policy agents at current timestep."""
        if not self.enabled:
            return

        timestep_data = {"timestep": self.timestep, "agents": []}

        # Process policy agents
        if policy_state.lstm_h is not None and policy_state.lstm_c is not None:
            # Concatenate hidden and cell states along dimension 0 to match expected shape
            # LSTM states are (num_layers, batch_size, hidden_size)
            # We want (2 * num_layers, batch_size, hidden_size)
            memory_vectors = torch.cat([policy_state.lstm_h, policy_state.lstm_c], dim=0)

            # Create a mapping from policy indices to agent IDs
            agent_id_map = {}
            for obj_id, obj in env_grid_objects.items():
                if obj.get("type") == 0:  # Agent type
                    agent_id = obj.get("agent_id")
                    if agent_id is not None:
                        agent_id_map[agent_id] = obj_id

            for i, agent_idx in enumerate(policy_idxs):
                agent_idx_int = int(agent_idx.item())
                # Get all layers for this agent and flatten to 1D
                # Shape: (2 * num_layers, hidden_size) -> (2 * num_layers * hidden_size,)
                memory_vector = memory_vectors[:, i].flatten().cpu()

                # Get agent position from grid objects using the mapping
                if agent_idx_int in agent_id_map:
                    grid_obj_id = agent_id_map[agent_idx_int]
                    grid_obj = env_grid_objects[grid_obj_id]
                    position = (grid_obj["r"], grid_obj["c"])
                else:
                    logger.warning(f"Agent {agent_idx_int} not found in grid objects")
                    continue

                timestep_data["agents"].append(
                    {
                        "agent_id": agent_idx_int,
                        "memory_vector": memory_vector.tolist(),
                        "position": position,
                    }
                )

        self.data.append(timestep_data)
        self.timestep += 1

    def save(self):
        """Save logged data to JSON file."""
        if not self.enabled or not self.data:
            return

        try:
            with open(self.output_file, "w") as f:
                json.dump(self.data, f)
            logger.info(f"Doxascope data saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save doxascope data: {e}")


def preprocess_doxascope_data(
    json_files: list,
    preprocessed_dir: Path,
    output_filename: str = "training_data.npz",
    num_future_timesteps: int = 1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Preprocess all doxascope JSON files to create training data.

    Args:
        json_files: List of JSON log files to process
        preprocessed_dir: Directory where the output NPZ file will be saved
        output_filename: Name of the output NPZ file
        num_future_timesteps: The number of future steps to predict

    Returns:
        Tuple of (X, y) arrays where X is memory vectors and y is movement classes
    """
    output_file = preprocessed_dir / output_filename

    if not json_files:
        logger.warning("No JSON files provided for preprocessing")
        return None, None

    memory_vectors = []
    movements = []
    expected_dim = None

    logger.info(f"Processing {len(json_files)} files...")

    # Process new files
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Get all unique agent IDs in this file
        agent_ids = set()
        for timestep_data in data:
            for agent in timestep_data["agents"]:
                agent_ids.add(agent["agent_id"])

        # Process each agent separately
        for agent_id in agent_ids:
            # Extract trajectory for this specific agent
            trajectory = []
            for timestep_data in data:
                for agent in timestep_data["agents"]:
                    if agent["agent_id"] == agent_id:
                        memory = np.array(agent["memory_vector"], dtype=np.float32)
                        # Set expected dimension from first memory vector if not set
                        if expected_dim is None:
                            expected_dim = memory.shape[0]
                        # Skip if memory vector has wrong dimension
                        elif memory.shape[0] != expected_dim:
                            logger.warning(
                                f"Skipping memory vector with dimension {memory.shape[0]} (expected {expected_dim})"
                            )
                            continue
                        position = agent["position"]
                        trajectory.append((memory, position))
                        break

            # Create training pairs from consecutive timesteps
            for i in range(len(trajectory) - num_future_timesteps):
                current_memory, start_pos = trajectory[i]
                future_movements = []

                # Previous position starts as the agent's current position at time `i`
                prev_pos = start_pos

                # Look ahead k steps
                for k in range(1, num_future_timesteps + 1):
                    _, next_pos = trajectory[i + k]

                    # Calculate relative movement from the PREVIOUS step
                    dr = next_pos[0] - prev_pos[0]  # row delta
                    dc = next_pos[1] - prev_pos[1]  # col delta

                    # Update previous position for the next iteration
                    prev_pos = next_pos

                    # Convert to movement class
                    if dr == 0 and dc == 0:
                        movement = 0  # stay
                    elif dr == -1 and dc == 0:
                        movement = 1  # up
                    elif dr == 1 and dc == 0:
                        movement = 2  # down
                    elif dr == 0 and dc == -1:
                        movement = 3  # left
                    elif dr == 0 and dc == 1:
                        movement = 4  # right
                    else:
                        # This case can happen if the agent teleports or moves > 1 cell
                        logger.warning(f"Unexpected movement: dr={dr}, dc={dc}. Defaulting to 'stay'.")
                        movement = 0  # default to 'stay'
                    future_movements.append(movement)

                memory_vectors.append(current_memory)
                movements.append(future_movements)

    if not memory_vectors:
        logger.warning("No training data created")
        return None, None

    try:
        X = np.array(memory_vectors, dtype=np.float32)
        y = np.array(movements, dtype=np.int64)

        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, X=X, y=y)
        logger.info(f"Successfully saved {len(X)} samples to {output_file}")
        return X, y
    except ValueError as e:
        logger.error(f"Failed to create arrays: {e}. The requested array has an inhomogeneous shape.")
        # Log detected shapes for debugging
        if memory_vectors:
            dims = {mv.shape[0] for mv in memory_vectors if hasattr(mv, "shape")}
            logger.error(f"Found memory vectors with dimensions: {dims}")
        return None, None
