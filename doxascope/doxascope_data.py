#!/usr/bin/env python3
"""
Doxascope Data Module

This module provides functionality for:
1. Logging LSTM memory vectors and agent positions during simulation using the DoxascopeLogger class.
2. Preprocessing the logged data for neural network training

"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from metta.agent.policy_state import PolicyState

logger = logging.getLogger(__name__)


# Coordinate Conversion Utilities
def get_positions_for_manhattan_distance(d: int) -> List[Tuple[int, int]]:
    """
    Returns a canonical, sorted list of all (dr, dc) positions
    within a given Manhattan distance.
    """
    d = abs(d)
    positions = []
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if abs(dr) + abs(dc) <= d:
                positions.append((dr, dc))
    # Sort by row, then column for a canonical order
    return sorted(positions)


def get_num_classes_for_manhattan_distance(d: int) -> int:
    """Returns the number of reachable cells within a given Manhattan distance."""
    d = abs(d)
    return 2 * d * d + 2 * d + 1


def get_pos_to_class_id_map(d: int) -> Dict[Tuple[int, int], int]:
    """Returns a mapping from (dr, dc) -> class_id for a given Manhattan distance."""
    positions = get_positions_for_manhattan_distance(d)
    return {pos: i for i, pos in enumerate(positions)}


def get_class_id_to_pos_map(d: int) -> Dict[int, Tuple[int, int]]:
    """Returns a mapping from class_id -> (dr, dc) for a given Manhattan distance."""
    positions = get_positions_for_manhattan_distance(d)
    return {i: pos for i, pos in enumerate(positions)}


def pos_to_class_id(dr: int, dc: int, d: int) -> int:
    """Converts a relative position (dr, dc) to a class index for a given Manhattan distance."""
    mapping = get_pos_to_class_id_map(d)
    pos = (dr, dc)
    if pos not in mapping:
        raise ValueError(f"Position ({dr}, {dc}) is outside max Manhattan distance {d}")
    return mapping[pos]


def class_id_to_pos(class_id: int, d: int) -> Tuple[int, int]:
    """Converts a class index back to a relative position (dr, dc) for a given Manhattan distance."""
    mapping = get_class_id_to_pos_map(d)
    if class_id not in mapping:
        raise ValueError(f"Class ID {class_id} is out of bounds for Manhattan distance {d}")
    return mapping[class_id]


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

        base_dir = Path(doxascope_config.get("output_dir", "./train_dir/doxascope/raw_data/"))
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


def _extract_agent_trajectories(files: list) -> Tuple[Dict[int, list], Optional[int]]:
    """Loads raw data from JSON files and organizes it into per-agent trajectories."""
    agent_trajectories: Dict[int, list] = {}
    expected_dim = None

    for json_file in files:
        with open(json_file, "r") as f:
            data = json.load(f)

        for timestep_data in data:
            for agent_data in timestep_data["agents"]:
                agent_id = agent_data["agent_id"]
                if agent_id not in agent_trajectories:
                    agent_trajectories[agent_id] = []

                memory = np.array(agent_data["memory_vector"], dtype=np.float32)

                if expected_dim is None:
                    expected_dim = memory.shape[0]
                elif memory.shape[0] != expected_dim:
                    logger.warning(
                        f"Skipping memory vector for agent {agent_id} in {json_file} with dimension "
                        f"{memory.shape[0]} (expected {expected_dim})"
                    )
                    continue

                position = tuple(agent_data["position"])
                agent_trajectories[agent_id].append((memory, position))

    return agent_trajectories, expected_dim


def _create_training_samples(
    agent_trajectories: Dict[int, list], num_future: int, num_past: int
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Generates training samples (X, y) from agent trajectories."""
    all_memory_vectors = []
    all_labels = []

    for trajectory in agent_trajectories.values():
        if len(trajectory) < num_future + num_past + 1:
            continue

        for i in range(num_past, len(trajectory) - num_future):
            current_memory, current_pos = trajectory[i]
            timestep_labels = []
            valid_sample = True

            # Create labels for past and future timesteps
            for k in list(range(-num_past, 0)) + list(range(1, num_future + 1)):
                pos_k = trajectory[i + k][1]
                dr, dc = pos_k[0] - current_pos[0], pos_k[1] - current_pos[1]
                max_dist = abs(k)

                if abs(dr) + abs(dc) > max_dist:
                    valid_sample = False
                    break  # Skip sample if any position is out of bounds

                label = pos_to_class_id(dr, dc, max_dist)
                timestep_labels.append(label)

            if valid_sample:
                all_labels.append(timestep_labels)
                all_memory_vectors.append(current_memory)

    return all_memory_vectors, all_labels


def preprocess_doxascope_data(
    json_files: list,
    preprocessed_dir: Path,
    output_filename: str = "training_data.npz",
    num_future_timesteps: int = 1,
    num_past_timesteps: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Preprocesses raw doxascope JSON data to create training-ready NPZ files.
    """
    if not json_files:
        logger.warning("No JSON files provided for preprocessing.")
        return None, None

    logger.info(f"Processing {len(json_files)} simulation log(s)...")

    agent_trajectories, expected_dim = _extract_agent_trajectories(json_files)
    if not agent_trajectories:
        logger.warning("No valid agent trajectories found in the provided files.")
        return None, None

    all_memory_vectors, all_labels = _create_training_samples(
        agent_trajectories, num_future_timesteps, num_past_timesteps
    )

    if not all_memory_vectors:
        logger.warning("No training samples could be created from the trajectories.")
        return None, None

    try:
        X = np.array(all_memory_vectors, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        output_file = preprocessed_dir / output_filename
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, X=X, y=y)

        logger.info(f"Successfully saved {len(X)} samples to {output_file}")
        return X, y
    except ValueError as e:
        logger.error(f"Failed to create NumPy arrays due to inconsistent shapes: {e}")
        unique_dims = {mv.shape for mv in all_memory_vectors}
        logger.error(f"Found memory vectors with the following shapes: {unique_dims}")
        return None, None
