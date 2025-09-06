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
    # Import genny-generated bindings (lowercase 'tribal', not 'Tribal')
    import tribal
    
    # Extract classes and functions
    TribalEnv = tribal.TribalEnv
    TribalConfig = tribal.TribalConfig
    TribalGameConfig = tribal.TribalGameConfig
    SeqInt = tribal.SeqInt
    SeqFloat = tribal.SeqFloat
    SeqBool = tribal.SeqBool
    
    # Constants
    MAP_AGENTS = tribal.MAP_AGENTS
    OBSERVATION_LAYERS = tribal.OBSERVATION_LAYERS  
    OBSERVATION_WIDTH = tribal.OBSERVATION_WIDTH
    OBSERVATION_HEIGHT = tribal.OBSERVATION_HEIGHT
    MAP_WIDTH = tribal.MAP_WIDTH
    MAP_HEIGHT = tribal.MAP_HEIGHT
    
    # Helper functions  
    default_max_steps = tribal.default_max_steps
    default_tribal_config = tribal.default_tribal_config
    check_error = tribal.check_error
    take_error = tribal.take_error
    
    # Constants mapping
    MapAgents = MAP_AGENTS
    ObservationLayers = OBSERVATION_LAYERS  
    ObservationWidth = OBSERVATION_WIDTH
    ObservationHeight = OBSERVATION_HEIGHT
except ImportError as e:
    raise ImportError(
        f"Could not import tribal bindings: {e}\n"
        f"Run 'cd mettascope2 && nimble bindings' to generate bindings."
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
        # Create Nim configuration object
        if config is None:
            nim_config = default_tribal_config()
        else:
            # Start with default configuration
            nim_config = default_tribal_config()
            
            # Override with provided config values (only configurable parameters)
            if 'max_steps' in config:
                nim_config.game.maxSteps = config['max_steps']
            if 'ore_per_battery' in config:
                nim_config.game.orePerBattery = config['ore_per_battery']
            if 'batteries_per_heart' in config:
                nim_config.game.batteriesPerHeart = config['batteries_per_heart']
            if 'enable_combat' in config:
                nim_config.game.enableCombat = config['enable_combat']
            if 'clippy_spawn_rate' in config:
                nim_config.game.clippySpawnRate = config['clippy_spawn_rate']
            if 'clippy_damage' in config:
                nim_config.game.clippyDamage = config['clippy_damage']
            if 'heart_reward' in config:
                nim_config.game.heartReward = config['heart_reward']
            if 'battery_reward' in config:
                nim_config.game.batteryReward = config['battery_reward']
            if 'ore_reward' in config:
                nim_config.game.oreReward = config['ore_reward']
            if 'survival_penalty' in config:
                nim_config.game.survivalPenalty = config['survival_penalty']
            if 'death_penalty' in config:
                nim_config.game.deathPenalty = config['death_penalty']
        
        # Create Nim environment instance with full configuration
        self._nim_env = TribalEnv(nim_config)
        self._config = nim_config
        
        # Cache dimensions (compile-time constants)
        self.num_agents = MapAgents
        self.observation_layers = ObservationLayers
        self.observation_width = ObservationWidth
        self.observation_height = ObservationHeight
        
        # Action space info (tribal has 6 action types, 8 directional arguments)
        self.num_action_types = 6  # NOOP, MOVE, ATTACK, GET, SWAP, PUT
        self.max_argument = 8      # 8-directional

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observations."""
        # Reset the Nim environment (our binding doesn't take seed parameter)
        self._nim_env.reset_env()
        
        # Get observations and convert to numpy
        obs_data = self._nim_env.get_observations()
        observations = self._convert_observations(obs_data)
        
        info = {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self._config.game.max_steps,
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
        
        # Convert actions to SeqInt expected by Nim
        actions_seq = SeqInt()
        for i in range(self.num_agents):
            actions_seq.append(int(actions[i, 0]))
            actions_seq.append(int(actions[i, 1]))
        
        # Step the environment
        success = self._nim_env.step(actions_seq)
        if not success:
            raise RuntimeError("Environment step failed")
        
        # Get results
        obs_data = self._nim_env.get_observations()
        observations = self._convert_observations(obs_data)
        
        rewards_seq = self._nim_env.get_rewards()
        rewards = np.array([rewards_seq[i] for i in range(len(rewards_seq))], dtype=np.float32)
        
        terminated_seq = self._nim_env.get_terminated()
        terminals = np.array([terminated_seq[i] for i in range(len(terminated_seq))], dtype=bool)
        
        truncated_seq = self._nim_env.get_truncated()
        truncations = np.array([truncated_seq[i] for i in range(len(truncated_seq))], dtype=bool)
        
        # Check for episode end
        if self._nim_env.is_episode_done():
            truncations[:] = True
        
        info = {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self._config.game.max_steps,
            "episode_done": self._nim_env.is_episode_done(),
        }
        
        return observations, rewards, terminals, truncations, info

    def _convert_observations(self, obs_seq: SeqInt) -> np.ndarray:
        """Convert genny observation data to numpy array."""
        # obs_seq is flattened: [agents * layers * height * width]
        obs_array = np.zeros(
            (self.num_agents, self.observation_layers, self.observation_height, self.observation_width),
            dtype=np.uint8
        )
        
        expected_size = self.num_agents * self.observation_layers * self.observation_height * self.observation_width
        
        # Convert SeqInt to numpy array and reshape
        if len(obs_seq) == expected_size:
            # Convert to python list then numpy and reshape
            obs_data = [obs_seq[i] for i in range(len(obs_seq))]
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
                            if index < len(obs_seq):
                                obs_array[agent_id, layer, y, x] = obs_seq[index]
                                index += 1
        
        return obs_array

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        if mode == "human":
            return self._nim_env.render_text()
        return None

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        # Our binding doesn't have get_episode_stats, return basic info
        return {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self._config.game.max_steps,
            "episode_done": self._nim_env.is_episode_done(),
        }

    @property
    def current_step(self) -> int:
        """Get current step."""
        return self._nim_env.get_current_step()

    @property
    def max_steps(self) -> int:
        """Get max steps."""
        return self._config.game.max_steps

    def close(self) -> None:
        """Clean up environment."""
        # Nim's GC will handle cleanup
        pass


def make_tribal_env(**config) -> TribalGridEnv:
    """
    Create a tribal environment instance using genny bindings.
    
    Args:
        **config: Configuration parameters for the environment
    
    Returns:
        TribalGridEnv instance
    """
    return TribalGridEnv(config)