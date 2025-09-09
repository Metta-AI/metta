"""
Tribal Environment using Genny-generated bindings.

This provides a clean Python interface to the Nim tribal environment
using genny's automatically generated bindings.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import Field

from metta.mettagrid.config import Config

# Add the genny-generated bindings to Python path
_BINDINGS_PATHS = [
    Path(__file__).parent.parent.parent / "tribal" / "bindings" / "generated",
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
        f"Could not import tribal bindings: {e}\nRun 'cd mettascope2 && nimble bindings' to generate bindings."
    ) from e


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
            if "max_steps" in config:
                nim_config.game.max_steps = config["max_steps"]
            if "ore_per_battery" in config:
                nim_config.game.ore_per_battery = config["ore_per_battery"]
            if "batteries_per_heart" in config:
                nim_config.game.batteries_per_heart = config["batteries_per_heart"]
            if "enable_combat" in config:
                nim_config.game.enable_combat = config["enable_combat"]
            if "clippy_spawn_rate" in config:
                nim_config.game.clippy_spawn_rate = config["clippy_spawn_rate"]
            if "clippy_damage" in config:
                nim_config.game.clippy_damage = config["clippy_damage"]
            if "heart_reward" in config:
                nim_config.game.heart_reward = config["heart_reward"]
            if "battery_reward" in config:
                nim_config.game.battery_reward = config["battery_reward"]
            if "ore_reward" in config:
                nim_config.game.ore_reward = config["ore_reward"]
            if "survival_penalty" in config:
                nim_config.game.survival_penalty = config["survival_penalty"]
            if "death_penalty" in config:
                nim_config.game.death_penalty = config["death_penalty"]

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
        self.max_argument = 8  # 8-directional
        
        # Add gym spaces for pufferlib compatibility
        import gymnasium as gym
        
        # Observation space: [layers, height, width]
        self.single_observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(self.observation_layers, self.observation_height, self.observation_width), 
            dtype=np.uint8
        )
        
        # Action space: [action_type, argument]  
        self.single_action_space = gym.spaces.MultiDiscrete([self.num_action_types, self.max_argument])
        
        # Cache for compatibility
        self.obs_width = self.observation_width
        self.obs_height = self.observation_height
        
        # Feature normalizations (create entries for each observation layer)
        # Tribal observations are raw pixel-like data, normalize to 0-255 range
        self.feature_normalizations = {i: 255.0 for i in range(self.observation_layers)}
        
        # Episode tracking for curriculum
        self._episode_rewards = np.array([0.0])

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observations."""
        # Reset the Nim environment (our binding doesn't take seed parameter)
        self._nim_env.reset_env()
        
        # Reset episode tracking
        self._episode_rewards[0] = 0.0

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

        # Track episode rewards for curriculum
        self._episode_rewards[0] += rewards.sum()

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
            (self.num_agents, self.observation_layers, self.observation_height, self.observation_width), dtype=np.uint8
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
    
    def set_mg_config(self, config) -> None:
        """Set new MettaGrid configuration (for curriculum compatibility)."""
        # For tribal environments, we don't need to change configuration during curriculum
        # as the environment configuration is compile-time
        pass
    
    def get_episode_rewards(self) -> np.ndarray:
        """Get episode rewards (for curriculum compatibility)."""
        return self._episode_rewards
    
    def get_observation_features(self):
        """Get observation features (for compatibility with MettaGrid interface)."""
        # Return empty dict as tribal observations don't use the same feature system
        return {}
    
    @property
    def action_names(self) -> list[str]:
        """Get action names for tribal environment."""
        return ["NOOP", "MOVE", "ATTACK", "GET", "SWAP", "PUT"]
    
    @property
    def max_action_args(self) -> list[int]:
        """Get maximum action arguments for each action type."""
        # For tribal actions: NOOP=0, MOVE=8, ATTACK=8, GET=8, SWAP=8, PUT=8
        # All directional actions use 8 directions, NOOP uses 0
        return [0, 8, 8, 8, 8, 8]

    def close(self) -> None:
        """Clean up environment."""
        # Nim's GC will handle cleanup
        pass


# Configuration Classes
class TribalGameConfig(Config):
    """Configuration for tribal game mechanics.

    NOTE: Structural parameters (num_agents, map dimensions, observation space)
    are kept as compile-time constants for performance. Only gameplay parameters
    are configurable at runtime.
    """

    # Core game parameters
    max_steps: int = Field(default=2000, ge=0, description="Maximum steps per episode")

    # Resource configuration
    ore_per_battery: int = Field(default=3, description="Ore required to craft battery")
    batteries_per_heart: int = Field(default=2, description="Batteries required at altar for hearts")

    # Combat configuration
    enable_combat: bool = Field(default=True, description="Enable agent combat")
    clippy_spawn_rate: float = Field(default=1.0, ge=0, le=1, description="Rate of enemy spawning")
    clippy_damage: int = Field(default=1, description="Damage dealt by enemies")

    # Reward configuration
    heart_reward: float = Field(default=1.0, description="Reward for creating hearts")
    ore_reward: float = Field(default=0.1, description="Reward for collecting ore")
    battery_reward: float = Field(default=0.8, description="Reward for crafting batteries")
    survival_penalty: float = Field(default=-0.01, description="Per-step survival penalty")
    death_penalty: float = Field(default=-5.0, description="Penalty for agent death")

    @property
    def num_agents(self) -> int:
        """Number of agents (compile-time constant)."""
        return MAP_AGENTS


class TribalEnvConfig(Config):
    """Configuration for Nim tribal environments."""

    environment_type: str = "tribal"
    label: str = Field(default="tribal", description="Environment label")

    # Game configuration
    game: TribalGameConfig = Field(default_factory=TribalGameConfig)

    # Environment settings
    desync_episodes: bool = Field(default=True, description="Desynchronize episode resets")
    render_mode: Optional[str] = Field(default=None, description="Rendering mode (human, rgb_array)")

    def get_observation_space(self) -> Dict[str, Any]:
        """Get tribal environment observation space."""
        return {
            "shape": (OBSERVATION_LAYERS, OBSERVATION_HEIGHT, OBSERVATION_WIDTH),
            "dtype": "uint8",
            "type": "Box",
        }

    def get_action_space(self) -> Dict[str, Any]:
        """Get tribal environment action space."""
        return {
            "shape": (2,),  # [action_type, argument]
            "dtype": "int32",
            "type": "MultiDiscrete",
            "nvec": [6, 8],  # 6 action types, 8 directions/targets
        }

    def create_environment(self, **kwargs) -> Any:
        """Create tribal environment instance."""
        # Convert configuration to dictionary format expected by make_tribal_env
        config = {
            "max_steps": self.game.max_steps,
            "ore_per_battery": self.game.ore_per_battery,
            "batteries_per_heart": self.game.batteries_per_heart,
            "enable_combat": self.game.enable_combat,
            "clippy_spawn_rate": self.game.clippy_spawn_rate,
            "clippy_damage": self.game.clippy_damage,
            "heart_reward": self.game.heart_reward,
            "battery_reward": self.game.battery_reward,
            "ore_reward": self.game.ore_reward,
            "survival_penalty": self.game.survival_penalty,
            "death_penalty": self.game.death_penalty,
            "render_mode": self.render_mode,
            **kwargs,
        }

        return make_tribal_env(**config)


def make_tribal_env(**config) -> TribalGridEnv:
    """
    Create a tribal environment instance using genny bindings.

    Args:
        **config: Configuration parameters for the environment

    Returns:
        TribalGridEnv instance
    """
    return TribalGridEnv(config)
