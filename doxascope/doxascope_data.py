#!/usr/bin/env python3
"""
Doxascope Data Module

This module provides functionality for:
1. Logging LSTM memory vectors and agent positions during simulation using the DoxascopeLogger class.
2. Preprocessing the logged data for neural network training

"""

import json
import logging
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from metta.agent.policy_state import PolicyState

logger = logging.getLogger(__name__)


class Movement(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


MOVEMENT_MAP = {
    (0, 0): Movement.STAY,
    (-1, 0): Movement.UP,
    (1, 0): Movement.DOWN,
    (0, -1): Movement.LEFT,
    (0, 1): Movement.RIGHT,
}


class DoxascopeLogger:
    """Logs memory vectors and position data for training doxascope networks."""

    def __init__(
        self,
        doxascope_config: dict,
        simulation_id: str,
        policy_name: str,
        object_type_names: Optional[List[str]] = None,
    ):
        self.enabled = doxascope_config.get("enabled", False)
        if not self.enabled:
            return

        base_dir = Path(doxascope_config.get("output_dir", "./doxascope/data/raw_data/"))
        self.output_dir = base_dir / policy_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_dir / f"doxascope_data_{simulation_id}.json"
        self.data = []
        self.timestep = 0
        self.agent_id_map: Optional[Dict[int, int]] = None

        if object_type_names and "agent" in object_type_names:
            self.agent_type_id = object_type_names.index("agent")
        else:
            self.agent_type_id = 0  # Default to 0 if not found
            logger.warning("Could not find 'agent' in object_type_names, defaulting to type ID 0.")

        logger.info(f"Doxascope logging enabled for policy '{policy_name}', will save raw data to {self.output_file}")

    def _build_agent_id_map(self, env_grid_objects: Dict) -> Dict[int, int]:
        """Builds a mapping from agent IDs to grid object IDs."""
        agent_id_map = {}
        for obj_id, obj in env_grid_objects.items():
            if obj.get("type") == self.agent_type_id:
                agent_id = obj.get("agent_id")
                if agent_id is not None:
                    agent_id_map[agent_id] = obj_id
        return agent_id_map

    def log_timestep(
        self,
        policy_state: PolicyState,
        policy_idxs: torch.Tensor,
        env_grid_objects: Dict,
    ):
        """Log memory vectors and positions for policy agents at current timestep."""
        if not self.enabled:
            return

        if self.agent_id_map is None:
            self.agent_id_map = self._build_agent_id_map(env_grid_objects)

        timestep_data = {"timestep": self.timestep, "agents": []}

        if policy_state.lstm_h is not None and policy_state.lstm_c is not None:
            memory_vectors = torch.cat([policy_state.lstm_h, policy_state.lstm_c], dim=0)

            for i, agent_idx in enumerate(policy_idxs):
                agent_idx_int = int(agent_idx.item())
                memory_vector = memory_vectors[:, i].flatten().cpu()

                if agent_idx_int in self.agent_id_map:
                    grid_obj_id = self.agent_id_map[agent_idx_int]
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
    num_past_timesteps: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Preprocess all doxascope JSON files to create training data.

    Args:
        json_files: List of JSON log files to process
        preprocessed_dir: Directory where the output NPZ file will be saved
        output_filename: Name of the output NPZ file
        num_future_timesteps: The number of future steps to predict
        num_past_timesteps: The number of past steps to predict

    Returns:
        A tuple containing two NumPy arrays:
        - X: The input data, with shape (num_samples, memory_vector_dim). Each row is a
          flattened LSTM memory vector.
        - y: The target data, with shape (num_samples, num_past_timesteps + num_future_timesteps).
          Each row contains the sequence of past and future movement classes to be predicted.
    """
    output_file = preprocessed_dir / output_filename

    if not json_files:
        logger.warning("No JSON files provided for preprocessing")
        return None, None

    all_memory_vectors = []
    all_movements = []
    expected_dim = None

    logger.info(f"Processing {len(json_files)} files...")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        agent_trajectories: Dict[int, list] = {}
        for timestep_data in data:
            for agent in timestep_data["agents"]:
                agent_id = agent["agent_id"]
                if agent_id not in agent_trajectories:
                    agent_trajectories[agent_id] = []

                memory = np.array(agent["memory_vector"], dtype=np.float32)

                if expected_dim is None:
                    expected_dim = memory.shape[0]
                elif memory.shape[0] != expected_dim:
                    logger.warning(
                        f"Skipping memory vector for agent {agent_id} with dimension {memory.shape[0]} (expected {expected_dim})"
                    )
                    continue

                position = agent["position"]
                agent_trajectories[agent_id].append((memory, position))

        for agent_id, trajectory in agent_trajectories.items():
            if len(trajectory) <= num_future_timesteps + num_past_timesteps + 1:
                continue

            for i in range(num_past_timesteps + 1, len(trajectory) - num_future_timesteps):
                current_memory, _ = trajectory[i]

                past_movements = []
                for k in range(num_past_timesteps, 0, -1):
                    if i - k - 1 < 0:
                        continue
                    pos_after = trajectory[i - k][1]
                    pos_before = trajectory[i - k - 1][1]
                    dr, dc = pos_after[0] - pos_before[0], pos_after[1] - pos_before[1]
                    past_movements.append(MOVEMENT_MAP.get((dr, dc), Movement.STAY))

                future_movements = []
                for k in range(1, num_future_timesteps + 1):
                    pos_after = trajectory[i + k][1]
                    pos_before = trajectory[i + k - 1][1]

                    dr, dc = pos_after[0] - pos_before[0], pos_after[1] - pos_before[1]
                    movement = MOVEMENT_MAP.get((dr, dc))
                    if movement is None:
                        logger.warning(
                            f"Unexpected movement for agent {agent_id}: dr={dr}, dc={dc}. Defaulting to 'stay'."
                        )
                        movement = Movement.STAY
                    future_movements.append(movement)

                all_movements.append(past_movements + future_movements)
                all_memory_vectors.append(current_memory)

    if not all_memory_vectors:
        logger.warning("No training data created")
        return None, None

    try:
        X = np.array(all_memory_vectors, dtype=np.float32)
        y = np.array(all_movements, dtype=np.int64)

        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, X=X, y=y)
        logger.info(f"Successfully saved {len(X)} samples to {output_file}")
        return X, y
    except ValueError as e:
        logger.error(f"Failed to create NumPy arrays due to inconsistent shapes: {e}")
        if all_memory_vectors:
            unique_dims = {mv.shape for mv in all_memory_vectors}
            logger.error(f"Found memory vectors with the following shapes: {unique_dims}")
        return None, None
