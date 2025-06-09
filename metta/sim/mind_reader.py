# metta/sim/mind_reader.py
"""
Mind reader logging functionality for capturing LSTM memory vectors and agent positions.

This module provides the MindReaderLogger class that logs agent memory states and positions
during simulation for training neural networks that predict agent trajectories from memory.
"""

import json
import logging
from pathlib import Path
from typing import Dict

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

        self.output_dir = Path(mind_reader_config.get("output_dir", "./mind_reader_data"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.simulation_id = simulation_id
        self.timestep = 0
        self.data_buffer = []

        logger.info(f"MindReaderLogger initialized for simulation {simulation_id}")

    def log_timestep(
        self,
        policy_state: PolicyState,
        npc_state: PolicyState,
        policy_idxs: torch.Tensor,
        npc_idxs: torch.Tensor,
        env_grid_objects: Dict,
    ):
        """Log memory vectors and positions for current timestep."""
        if not self.enabled:
            return

        timestep_data = {"timestep": self.timestep, "agents": []}

        # Log policy agent data
        if policy_state.lstm_h is not None and policy_state.lstm_c is not None:
            for i, agent_idx in enumerate(policy_idxs):
                agent_idx_int = int(agent_idx.item())

                # Extract memory vector (concatenate h and c states for this agent)
                h_state = policy_state.lstm_h[:, i, :].cpu().numpy().flatten()  # Shape: (num_layers * hidden_size,)
                c_state = policy_state.lstm_c[:, i, :].cpu().numpy().flatten()  # Shape: (num_layers * hidden_size,)
                memory_vector = np.concatenate([h_state, c_state])

                # Find agent position from grid objects
                position = self._find_agent_position(agent_idx_int, env_grid_objects)

                timestep_data["agents"].append(
                    {
                        "agent_id": agent_idx_int,
                        "agent_type": "policy",
                        "memory_vector": memory_vector.tolist(),
                        "position": position,
                    }
                )

        # Log NPC agent data if present
        if len(npc_idxs) > 0 and npc_state.lstm_h is not None and npc_state.lstm_c is not None:
            for i, agent_idx in enumerate(npc_idxs):
                agent_idx_int = int(agent_idx.item())

                # Extract memory vector
                h_state = npc_state.lstm_h[:, i, :].cpu().numpy().flatten()
                c_state = npc_state.lstm_c[:, i, :].cpu().numpy().flatten()
                memory_vector = np.concatenate([h_state, c_state])

                # Find agent position
                position = self._find_agent_position(agent_idx_int, env_grid_objects)

                timestep_data["agents"].append(
                    {
                        "agent_id": agent_idx_int,
                        "agent_type": "npc",
                        "memory_vector": memory_vector.tolist(),
                        "position": position,
                    }
                )

        self.data_buffer.append(timestep_data)
        self.timestep += 1

    def _find_agent_position(self, agent_id: int, grid_objects: Dict) -> tuple:
        """Find agent position from grid objects."""
        for obj_data in grid_objects.values():
            if obj_data.get("agent_id") == agent_id:
                return (int(obj_data["r"]), int(obj_data["c"]))
        return (-1, -1)  # Not found

    def save_data(self):
        """Save logged data to file."""
        if not self.enabled or not self.data_buffer:
            return

        output_file = self.output_dir / f"mind_reader_data_{self.simulation_id}.json"

        with open(output_file, "w") as f:
            json.dump(self.data_buffer, f, indent=2)

        logger.info(f"Saved {len(self.data_buffer)} timesteps of mind reader data to {output_file}")
        self.data_buffer = []
