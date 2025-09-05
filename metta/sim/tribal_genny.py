"""
Tribal Environment using Genny-generated bindings.

This provides a clean Python interface to the Nim tribal environment
using genny's automatically generated bindings.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Add the genny-generated bindings to Python path
_BINDINGS_PATHS = [
    Path(__file__).parent.parent.parent / "mettascope2" / "bindings" / "generated",
]

for path in _BINDINGS_PATHS:
    if path.exists():
        sys.path.insert(0, str(path))
        break

try:
    # Import genny-generated bindings
    from Tribal import (
        TribalEnv, TribalConfig, 
        newTribalEnv, defaultConfig, getActionSpace,
        MapAgents, ObservationLayers, ObservationWidth, ObservationHeight
    )
except ImportError as e:
    raise ImportError(
        f"Could not import Tribal bindings: {e}\n"
        f"Run 'cd mettascope2 && ./build_bindings.sh' to generate bindings."
    )

class TribalGridEnv:
    """
    Python wrapper for Nim tribal environment using genny bindings.
    
    Provides a clean, NumPy-based interface that's compatible with
    the training infrastructure while using the high-performance
    genny-generated bindings underneath.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tribal environment."""
        # Create configuration
        if config is None:
            self._nim_config = defaultConfig()
        else:
            nim_config = defaultConfig()
            nim_config.numAgents = config.get('num_agents', MapAgents)
            nim_config.maxSteps = config.get('max_steps', 1000)
            nim_config.mapWidth = config.get('map_width', 96) 
            nim_config.mapHeight = config.get('map_height', 46)
            nim_config.seed = config.get('seed', 0)
            self._nim_config = nim_config
        
        # Create Nim environment instance
        self._nim_env = newTribalEnv(self._nim_config)
        
        # Cache dimensions
        self.num_agents = MapAgents
        self.observation_layers = ObservationLayers
        self.observation_width = ObservationWidth
        self.observation_height = ObservationHeight
        
        # Action space info
        action_space_info = getActionSpace()
        self.num_action_types = action_space_info[0]
        self.max_argument = action_space_info[1]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observations."""
        # Reset the Nim environment
        if seed is not None:
            self._nim_env.resetEnv(seed)
        else:
            self._nim_env.resetEnv()
        
        # Get observations and convert to numpy
        obs_data = self._nim_env.getObservations()
        observations = self._convert_observations(obs_data)
        
        info = {
            "current_step": self._nim_env.getCurrentStep(),
            "max_steps": self._nim_env.getMaxSteps(),
        }
        
        return observations, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step environment with actions.
        
        Args:
            actions: np.ndarray of shape [num_agents, 2] with dtype int32
                    actions[:, 0] = action_type (0-5)
                    actions[:, 1] = argument (direction, target, etc.)
        
        Returns:
            observations: np.ndarray [num_agents, layers, height, width] uint8
            rewards: np.ndarray [num_agents] float32  
            terminals: np.ndarray [num_agents] bool
            truncations: np.ndarray [num_agents] bool
            info: dict with episode information
        """
        # Validate input
        if actions.shape != (self.num_agents, 2):
            raise ValueError(f"Actions must have shape ({self.num_agents}, 2), got {actions.shape}")
        
        # Convert actions to flat format expected by Nim
        actions_flat = []
        for i in range(self.num_agents):
            actions_flat.append(int(actions[i, 0]))
            actions_flat.append(int(actions[i, 1]))
        
        # Step the environment
        success = self._nim_env.step(actions_flat)
        if not success:
            raise RuntimeError("Environment step failed")
        
        # Get results
        obs_data = self._nim_env.getObservations()
        observations = self._convert_observations(obs_data)
        
        rewards_list = self._nim_env.getRewards()
        rewards = np.array(rewards_list, dtype=np.float32)
        
        terminated_list = self._nim_env.getTerminated()
        terminals = np.array(terminated_list, dtype=bool)
        
        truncated_list = self._nim_env.getTruncated()
        truncations = np.array(truncated_list, dtype=bool)
        
        # Check for episode end
        if self._nim_env.isEpisodeDone():
            truncations[:] = True
        
        info = {
            "current_step": self._nim_env.getCurrentStep(),
            "max_steps": self._nim_env.getMaxSteps(),
            "episode_done": self._nim_env.isEpisodeDone(),
        }
        
        return observations, rewards, terminals, truncations, info

    def _convert_observations(self, obs_data: List[int]) -> np.ndarray:
        """Convert genny observation data to numpy array."""
        # obs_data is flattened: [agents * layers * height * width]
        obs_array = np.zeros(
            (self.num_agents, self.observation_layers, self.observation_height, self.observation_width),
            dtype=np.uint8
        )
        
        # Reshape the flat array
        if len(obs_data) == self.num_agents * self.observation_layers * self.observation_height * self.observation_width:
            # Convert to numpy and reshape
            flat_array = np.array(obs_data, dtype=np.uint8)
            obs_array = flat_array.reshape(
                (self.num_agents, self.observation_layers, self.observation_height, self.observation_width)
            )
        else:
            # Fallback: fill array element by element (slower)
            index = 0
            for agent_id in range(self.num_agents):
                for layer in range(self.observation_layers):
                    for y in range(self.observation_height):
                        for x in range(self.observation_width):
                            if index < len(obs_data):
                                obs_array[agent_id, layer, y, x] = obs_data[index]
                                index += 1
        
        return obs_array

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        if mode == "human":
            return self._nim_env.renderText()
        return None

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        stats_text = self._nim_env.getEpisodeStats()
        return {"stats_text": stats_text}

    @property
    def current_step(self) -> int:
        """Get current step."""
        return self._nim_env.getCurrentStep()

    @property
    def max_steps(self) -> int:
        """Get max steps."""
        return self._nim_env.getMaxSteps()

    def close(self) -> None:
        """Clean up environment."""
        # Nim's GC will handle cleanup
        pass


def make_tribal_env(
    num_agents: int = 15,
    max_steps: int = 1000,
    **kwargs
) -> TribalGridEnv:
    """
    Create a tribal environment instance using genny bindings.
    
    Args:
        num_agents: Number of agents (fixed at compile time, but configurable in future)
        max_steps: Maximum steps per episode
        **kwargs: Additional configuration parameters
    
    Returns:
        TribalGridEnv instance
    """
    config = {
        'num_agents': num_agents,
        'max_steps': max_steps,
        **kwargs
    }
    
    return TribalGridEnv(config)