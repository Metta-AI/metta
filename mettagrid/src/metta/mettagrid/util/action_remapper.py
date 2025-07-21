"""Action remapping utilities for converting between movement modes."""

import numpy as np
from typing import Tuple, List, Dict, Optional


class TankToCardinalRemapper:
    """Converts tank-style actions (move forward/backward/rotate) to cardinal movements.
    
    This is used for backward compatibility when switching from relative to cardinal movement mode.
    """
    
    def __init__(self, action_names: List[str]):
        """Initialize the remapper with the list of action names.
        
        Args:
            action_names: List of action names in order (e.g., ["noop", "move", "rotate", ...])
        """
        self.action_names = action_names
        self.action_indices = {name: i for i, name in enumerate(action_names)}
        
        # Cache action indices
        self.move_idx = self.action_indices.get("move", -1)
        self.rotate_idx = self.action_indices.get("rotate", -1)
        
        # In cardinal mode, move action arguments map to directions
        # 0 = North (Up), 1 = South (Down), 2 = West (Left), 3 = East (Right)
        self.direction_to_cardinal = {
            0: 0,  # Up -> North
            1: 1,  # Down -> South  
            2: 2,  # Left -> West
            3: 3,  # Right -> East
        }
    
    def remap_actions(self, actions: np.ndarray, agent_orientations: Dict[int, int]) -> np.ndarray:
        """Convert tank-style actions to cardinal movements.
        
        Args:
            actions: Array of shape (num_agents, 2) with [action_type, action_arg]
            agent_orientations: Dict mapping agent_idx to current orientation (0=Up, 1=Down, 2=Left, 3=Right)
            
        Returns:
            Remapped actions array of same shape
        """
        remapped = actions.copy()
        num_agents = actions.shape[0]
        
        for agent_idx in range(num_agents):
            action_type = actions[agent_idx, 0]
            action_arg = actions[agent_idx, 1]
            
            # Handle rotate actions - convert to appropriate cardinal move
            if action_type == self.rotate_idx and self.rotate_idx >= 0:
                # In tank mode, rotate sets orientation directly
                # Convert this to a cardinal move in that direction
                target_direction = action_arg  # 0=Up, 1=Down, 2=Left, 3=Right
                remapped[agent_idx, 0] = self.move_idx
                remapped[agent_idx, 1] = self.direction_to_cardinal[target_direction]
                
                # Update tracked orientation
                agent_orientations[agent_idx] = target_direction
                
            # Handle move actions - convert based on current orientation
            elif action_type == self.move_idx and self.move_idx >= 0:
                current_orientation = agent_orientations.get(agent_idx, 0)  # Default to Up
                
                if action_arg == 0:  # Move forward
                    # Move in current facing direction
                    remapped[agent_idx, 1] = self.direction_to_cardinal[current_orientation]
                elif action_arg == 1:  # Move backward
                    # Move opposite to facing direction
                    opposite_orientations = {0: 1, 1: 0, 2: 3, 3: 2}  # Up<->Down, Left<->Right
                    opposite = opposite_orientations[current_orientation]
                    remapped[agent_idx, 1] = self.direction_to_cardinal[opposite]
        
        return remapped


class CardinalToTankRemapper:
    """Converts cardinal movements to tank-style actions.
    
    This can be used when a policy trained with cardinal movements needs to work
    in an environment configured for tank-style controls.
    """
    
    def __init__(self, action_names: List[str]):
        """Initialize the remapper with the list of action names."""
        self.action_names = action_names
        self.action_indices = {name: i for i, name in enumerate(action_names)}
        
        self.move_idx = self.action_indices.get("move", -1)
        self.rotate_idx = self.action_indices.get("rotate", -1)
        self.noop_idx = self.action_indices.get("noop", 0)
    
    def remap_actions(self, actions: np.ndarray, agent_orientations: Dict[int, int]) -> np.ndarray:
        """Convert cardinal movements to tank-style actions.
        
        Args:
            actions: Array of shape (num_agents, 2) with [action_type, action_arg]
            agent_orientations: Dict mapping agent_idx to current orientation
            
        Returns:
            Remapped actions array that may have more timesteps due to rotation requirements
        """
        # This is more complex as it may require multiple actions (rotate then move)
        # For simplicity, we'll convert each cardinal move to appropriate tank actions
        remapped = []
        num_agents = actions.shape[0]
        
        for agent_idx in range(num_agents):
            action_type = actions[agent_idx, 0]
            action_arg = actions[agent_idx, 1]
            
            if action_type == self.move_idx and self.move_idx >= 0:
                # Cardinal move - need to check if rotation is needed
                target_direction = action_arg  # 0=North, 1=South, 2=West, 3=East
                current_orientation = agent_orientations.get(agent_idx, 0)
                
                if target_direction == current_orientation:
                    # Already facing the right way, just move forward
                    remapped.append([agent_idx, self.move_idx, 0])
                elif self._is_opposite(target_direction, current_orientation):
                    # Facing opposite direction, move backward
                    remapped.append([agent_idx, self.move_idx, 1])
                else:
                    # Need to rotate first
                    remapped.append([agent_idx, self.rotate_idx, target_direction])
                    agent_orientations[agent_idx] = target_direction
                    # Then move forward in next timestep
                    remapped.append([agent_idx, self.move_idx, 0])
            else:
                # Non-movement action, pass through
                remapped.append([agent_idx, action_type, action_arg])
        
        return remapped
    
    def _is_opposite(self, dir1: int, dir2: int) -> bool:
        """Check if two directions are opposite."""
        opposites = {(0, 1), (1, 0), (2, 3), (3, 2)}
        return (dir1, dir2) in opposites