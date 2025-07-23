"""Scripted NPC implementations for dual-policy training."""

import logging
import random
from typing import Optional

import torch
from torch import Tensor

from metta.rl.trainer_config import ScriptedNPCConfig

logger = logging.getLogger(__name__)


class ScriptedNPC:
    """Base class for scripted NPC behaviors."""

    def __init__(self, config: ScriptedNPCConfig, num_agents: int, device: torch.device):
        self.config = config
        self.num_agents = num_agents
        self.device = device

        # Per-agent state tracking
        self.agent_states = {}
        self._initialize_agent_states()

    def _initialize_agent_states(self):
        """Initialize state for each agent."""
        for agent_id in range(self.num_agents):
            self.agent_states[agent_id] = {
                "direction": random.choice([0, 1, 2, 3]),  # 0=up, 1=right, 2=down, 3=left
                "last_position": None,
                "stuck_count": 0,
                "grid_search_state": None,
            }

    def get_actions(self, observations: Tensor) -> Tensor:
        """Generate actions for all agents based on their observations."""
        batch_size = observations.shape[0]
        actions = torch.zeros(batch_size, 2, device=self.device, dtype=torch.int32)

        for agent_id in range(batch_size):
            obs = observations[agent_id]
            action = self._get_single_agent_action(agent_id, obs)
            actions[agent_id] = action

        return actions

    def _get_single_agent_action(self, agent_id: int, observation: Tensor) -> Tensor:
        """Generate action for a single agent."""
        if self.config.type == "roomba":
            return self._roomba_action(agent_id, observation)
        elif self.config.type == "grid_search":
            return self._grid_search_action(agent_id, observation)
        else:
            raise ValueError(f"Unknown scripted NPC type: {self.config.type}")

    def _roomba_action(self, agent_id: int, observation: Tensor) -> Tensor:
        """Roomba behavior: move in consistent direction, turn when hitting walls."""
        # Extract current position and direction from observation
        # This is a simplified implementation - in practice, you'd need to parse the observation
        # to get the agent's current position and detect walls/obstacles

        state = self.agent_states[agent_id]
        current_direction = state["direction"]

        # Check if we should approach items
        if self.config.approach_items:
            item_action = self._try_approach_items(agent_id, observation)
            if item_action is not None:
                return item_action

        # Simple roomba logic: try to move forward, turn if stuck
        # For now, we'll use a simple random movement pattern
        # In a real implementation, you'd parse the observation to detect walls

        # Random movement with bias toward current direction
        if random.random() < 0.7:  # 70% chance to continue in current direction
            action_type = 0  # move
            action_arg = current_direction
        else:
            # Turn
            if self.config.roomba_direction == "clockwise":
                new_direction = (current_direction + 1) % 4
            else:
                new_direction = (current_direction - 1) % 4

            state["direction"] = new_direction
            action_type = 0  # move
            action_arg = new_direction

        return torch.tensor([action_type, action_arg], device=self.device, dtype=torch.int32)

    def _grid_search_action(self, agent_id: int, observation: Tensor) -> Tensor:
        """Grid search behavior: systematic exploration pattern."""
        state = self.agent_states[agent_id]

        # Check if we should approach items
        if self.config.approach_items:
            item_action = self._try_approach_items(agent_id, observation)
            if item_action is not None:
                return item_action

        # Initialize grid search state if needed
        if state["grid_search_state"] is None:
            state["grid_search_state"] = {
                "pattern": self.config.grid_search_pattern,
                "current_step": 0,
                "visited_positions": set(),
            }

        # Generate action based on pattern
        if self.config.grid_search_pattern == "spiral":
            return self._spiral_action(agent_id, observation)
        elif self.config.grid_search_pattern == "snake":
            return self._snake_action(agent_id, observation)
        elif self.config.grid_search_pattern == "random":
            return self._random_action(agent_id, observation)
        else:
            raise ValueError(f"Unknown grid search pattern: {self.config.grid_search_pattern}")

    def _spiral_action(self, agent_id: int, observation: Tensor) -> Tensor:
        """Spiral exploration pattern."""
        # Simplified spiral implementation
        # In practice, you'd track the spiral pattern more carefully
        state = self.agent_states[agent_id]
        search_state = state["grid_search_state"]

        # Simple spiral: alternate between moving forward and turning
        if search_state["current_step"] % 10 < 8:  # Move forward for 8 steps
            action_type = 0  # move
            action_arg = state["direction"]
        else:  # Turn every 2 steps
            state["direction"] = (state["direction"] + 1) % 4
            action_type = 0  # move
            action_arg = state["direction"]

        search_state["current_step"] += 1
        return torch.tensor([action_type, action_arg], device=self.device, dtype=torch.int32)

    def _snake_action(self, agent_id: int, observation: Tensor) -> Tensor:
        """Snake exploration pattern."""
        state = self.agent_states[agent_id]
        search_state = state["grid_search_state"]

        # Snake pattern: move in one direction, turn at edges
        # Simplified implementation
        if search_state["current_step"] % 20 < 18:  # Move forward for 18 steps
            action_type = 0  # move
            action_arg = state["direction"]
        else:  # Turn every 2 steps
            state["direction"] = (state["direction"] + 2) % 4  # Turn 180 degrees
            action_type = 0  # move
            action_arg = state["direction"]

        search_state["current_step"] += 1
        return torch.tensor([action_type, action_arg], device=self.device, dtype=torch.int32)

    def _random_action(self, agent_id: int, observation: Tensor) -> Tensor:
        """Random exploration pattern."""
        state = self.agent_states[agent_id]

        # Random movement with some consistency
        if random.random() < 0.8:  # 80% chance to continue in current direction
            action_type = 0  # move
            action_arg = state["direction"]
        else:
            # Random turn
            state["direction"] = random.randint(0, 3)
            action_type = 0  # move
            action_arg = state["direction"]

        return torch.tensor([action_type, action_arg], device=self.device, dtype=torch.int32)

    def _try_approach_items(self, agent_id: int, observation: Tensor) -> Optional[Tensor]:
        """Try to approach items in the agent's field of view."""
        # This is a placeholder implementation
        # In practice, you'd parse the observation to detect items
        # and calculate the direction to move toward them

        # For now, we'll use a simple heuristic: if we see an item, move toward it
        # This would need to be implemented based on the actual observation format

        # Placeholder: 10% chance to "see" an item and approach it
        if random.random() < 0.1:
            # Random direction toward "item"
            direction = random.randint(0, 3)
            return torch.tensor([0, direction], device=self.device, dtype=torch.int32)

        return None


def create_scripted_npc(config: ScriptedNPCConfig, num_agents: int, device: torch.device) -> ScriptedNPC:
    """Factory function to create a scripted NPC."""
    return ScriptedNPC(config, num_agents, device)
