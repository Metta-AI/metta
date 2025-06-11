#!/usr/bin/env python3
"""
Mind Reader Module

This module provides functionality for:
1. Logging LSTM memory vectors and agent positions during simulation
2. Preprocessing the logged data for neural network training

The goal is to train a "mind reader" neural network that can predict agent movement
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


class MindReaderLogger:
    """Logs memory vectors and position data for training mind reader networks."""

    def __init__(self, mind_reader_config, simulation_id: str):
        self.enabled = mind_reader_config.get("enabled", False)
        if not self.enabled:
            return

        # Use the new directory structure
        self.output_dir = Path(mind_reader_config.get("output_dir", "./mind_reader/data/raw_data/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up paths for both raw and preprocessed data
        self.raw_data_dir = self.output_dir
        self.preprocessed_dir = self.output_dir.parent / "preprocessed_data"
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = self.raw_data_dir / f"mind_reader_data_{simulation_id}.json"
        self.data = []
        self.timestep = 0

        logger.info(f"Mind reader logging enabled, will save to {self.output_file}")

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
            # Concatenate hidden and cell states
            memory_vectors = torch.cat([policy_state.lstm_h, policy_state.lstm_c], dim=-1)

            for i, agent_idx in enumerate(policy_idxs):
                agent_idx_int = int(agent_idx.item())
                memory_vector = memory_vectors[i].cpu()

                # Get agent position from grid objects
                if agent_idx_int in env_grid_objects:
                    position = env_grid_objects[agent_idx_int].pos
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
            logger.info(f"Mind reader data saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save mind reader data: {e}")

    def preprocess_data(
        self, output_filename: str = "training_data.npz"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Preprocess all mind reader JSON files to create training data.

        Args:
            output_filename: Name of the output NPZ file

        Returns:
            Tuple of (X, y) arrays where X is memory vectors and y is movement classes
        """
        if not self.enabled:
            logger.info("Mind reader not enabled, skipping preprocessing")
            return None, None

        output_file = self.preprocessed_dir / output_filename
        json_files = list(self.raw_data_dir.glob("mind_reader_data_*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {self.raw_data_dir}")
            return None, None

        memory_vectors = []
        movements = []

        logger.info(f"Processing {len(json_files)} files...")

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
                            position = agent["position"]
                            trajectory.append((memory, position))
                            break

                # Create training pairs from consecutive timesteps
                for i in range(len(trajectory) - 1):
                    current_memory, current_pos = trajectory[i]
                    _, next_pos = trajectory[i + 1]

                    # Calculate relative movement
                    dr = next_pos[0] - current_pos[0]  # row delta
                    dc = next_pos[1] - current_pos[1]  # col delta

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
                        continue  # skip multi-step movements

                    memory_vectors.append(current_memory)
                    movements.append(movement)

        if not memory_vectors:
            logger.warning("No training data created")
            return None, None

        # Convert to arrays and save
        X = np.array(memory_vectors)
        y = np.array(movements)

        logger.info(f"Created {len(X)} training samples")
        logger.info(f"Memory vector shape: {X.shape}")
        logger.info(f"Movement distribution: {np.bincount(y)}")

        np.savez_compressed(output_file, X=X, y=y)
        logger.info(f"Saved to {output_file}")

        return X, y
