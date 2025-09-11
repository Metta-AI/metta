"""
Simplified Tribal Environment using direct pointer access for zero-copy performance.

This provides a clean Python interface to the Nim tribal environment
using direct memory sharing via pointers instead of data conversion.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import Field

from metta.mettagrid.config import Config

# Add the genny-generated bindings to Python path
_BINDINGS_PATHS = [
    Path(__file__).parent.parent / "bindings" / "generated",  # tribal/src/../bindings/generated
]

for path in _BINDINGS_PATHS:
    if path.exists():
        sys.path.insert(0, str(path))
        break

try:
    # Import genny-generated bindings
    import importlib.util

    bindings_file = Path(__file__).parent.parent / "bindings" / "generated" / "tribal.py"
    spec = importlib.util.spec_from_file_location("tribal_bindings", bindings_file)
    tribal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tribal)

    # Extract what we need
    TribalEnv = tribal.TribalEnv
    TribalConfig = tribal.TribalConfig
    TribalGameConfig = tribal.TribalGameConfig
    default_tribal_config = tribal.default_tribal_config

    # Constants
    MAP_AGENTS = tribal.MAP_AGENTS
    OBSERVATION_LAYERS = tribal.OBSERVATION_LAYERS
    OBSERVATION_WIDTH = tribal.OBSERVATION_WIDTH
    OBSERVATION_HEIGHT = tribal.OBSERVATION_HEIGHT
    MAP_WIDTH = tribal.MAP_WIDTH
    MAP_HEIGHT = tribal.MAP_HEIGHT
    MAX_TOKENS_PER_AGENT = tribal.MAX_TOKENS_PER_AGENT
    NUM_ACTION_TYPES = tribal.NUM_ACTION_TYPES

    # Functions we need
    get_action_names = tribal.get_action_names
    get_max_action_args = tribal.get_max_action_args
    get_feature_normalizations = tribal.get_feature_normalizations
    is_emulated = tribal.is_emulated
    is_done = tribal.is_done

except ImportError as e:
    raise ImportError(
        f"Could not import tribal bindings: {e}\nRun 'cd tribal && ./build_bindings.sh' to generate bindings."
    ) from e


class TribalGridEnv:
    """
    Simplified Python wrapper for Nim tribal environment using direct pointer access.

    This provides zero-copy performance by sharing memory directly between Python and Nim.
    All numpy arrays are pre-allocated and Nim reads/writes directly to their memory.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None, buf: Optional[Any] = None
    ):
        """Initialize tribal environment with pre-allocated numpy arrays."""

        # Set environment variable to signal Python training mode
        os.environ["TRIBAL_PYTHON_CONTROL"] = "1"

        # Create Nim configuration
        if config is None:
            nim_config = default_tribal_config()
        else:
            nim_config = default_tribal_config()
            # Override with provided config values
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

        # Create Nim environment instance
        self._nim_env = TribalEnv(nim_config)
        self._config = nim_config

        # Pre-allocate all numpy arrays for zero-copy communication
        self.observations = np.zeros((MAP_AGENTS, MAX_TOKENS_PER_AGENT, 3), dtype=np.uint8)
        self.actions = np.zeros((MAP_AGENTS, 2), dtype=np.uint8)
        self.rewards = np.zeros(MAP_AGENTS, dtype=np.float32)
        self.terminals = np.zeros(MAP_AGENTS, dtype=bool)
        self.truncations = np.zeros(MAP_AGENTS, dtype=bool)

        # Cache environment properties
        self.num_agents = MAP_AGENTS
        self.observation_layers = OBSERVATION_LAYERS
        self.observation_width = OBSERVATION_WIDTH
        self.observation_height = OBSERVATION_HEIGHT
        self.max_steps = nim_config.game.max_steps

        # Set up gym spaces for compatibility
        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(MAX_TOKENS_PER_AGENT, 3), dtype=np.uint8)
        self.single_action_space = gym.spaces.MultiDiscrete([NUM_ACTION_TYPES, 8])

        # Additional properties for compatibility
        self.obs_width = self.observation_width
        self.obs_height = self.observation_height
        self.height = MAP_HEIGHT
        self.width = MAP_WIDTH
        self.grid_objects = {}
        self.render_mode = render_mode

        # Feature normalizations
        feature_norms_seq = get_feature_normalizations()
        self.feature_normalizations = {i: feature_norms_seq[i] for i in range(len(feature_norms_seq))}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment using direct pointer access."""
        # Get pointer to observations array as integer
        obs_ptr_int = self.observations.ctypes.data_as(ctypes.c_void_p).value or 0

        # Reset environment and get observations directly written to our array
        success = tribal.reset_and_get_obs_pointer(self._nim_env, obs_ptr_int)
        if not success:
            raise RuntimeError("Environment reset failed")

        info = {
            "current_step": tribal.get_current_step(self._nim_env),
            "max_steps": self._config.game.max_steps,
        }

        return self.observations, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step environment using direct pointer access for zero-copy performance."""
        # Validate input
        if actions.shape != (self.num_agents, 2):
            raise ValueError(f"Actions must have shape ({self.num_agents}, 2), got {actions.shape}")

        # Copy actions to our pre-allocated array
        self.actions[:] = actions.astype(np.uint8)

        # Get pointers to all our arrays as integers
        actions_ptr_int = self.actions.ctypes.data_as(ctypes.c_void_p).value or 0
        obs_ptr_int = self.observations.ctypes.data_as(ctypes.c_void_p).value or 0
        rewards_ptr_int = self.rewards.ctypes.data_as(ctypes.c_void_p).value or 0
        terminals_ptr_int = self.terminals.ctypes.data_as(ctypes.c_void_p).value or 0
        truncations_ptr_int = self.truncations.ctypes.data_as(ctypes.c_void_p).value or 0

        # Step environment - Nim reads actions and writes results directly to our arrays
        success = tribal.step_with_pointers(
            self._nim_env, actions_ptr_int, obs_ptr_int, rewards_ptr_int, terminals_ptr_int, truncations_ptr_int
        )

        if not success:
            raise RuntimeError("Environment step failed")

        info = {
            "current_step": tribal.get_current_step(self._nim_env),
            "max_steps": self._config.game.max_steps,
            "episode_done": tribal.is_episode_done(self._nim_env),
        }

        return self.observations, self.rewards, self.terminals, self.truncations, info

    # PufferLib compatibility properties
    @property
    def emulated(self) -> bool:
        """Native envs do not use emulation."""
        return is_emulated()

    @property
    def done(self) -> bool:
        """Check if environment is done."""
        return is_done(self._nim_env)

    @property
    def current_step(self) -> int:
        """Get current step."""
        return tribal.get_current_step(self._nim_env)

    @property
    def action_names(self) -> list[str]:
        """Return the names of all available actions."""
        names_seq = get_action_names()
        return [names_seq[i] for i in range(len(names_seq))]

    @property
    def max_action_args(self) -> list[int]:
        """Return the maximum argument values for each action type."""
        args_seq = get_max_action_args()
        return [args_seq[i] for i in range(len(args_seq))]

    def get_observation_features(self) -> dict[str, dict]:
        """Build the features dictionary for initialize_to_environment."""
        features = {}
        feature_norms_seq = get_feature_normalizations()
        for layer_id in range(len(feature_norms_seq)):
            features[f"layer_{layer_id}"] = {
                "id": layer_id,
                "normalization": feature_norms_seq[layer_id],
            }
        return features

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        if mode == "human":
            return tribal.render_text(self._nim_env)
        return None

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        return {
            "current_step": tribal.get_current_step(self._nim_env),
            "max_steps": self._config.game.max_steps,
            "episode_done": tribal.is_episode_done(self._nim_env),
        }

    def set_mg_config(self, config) -> None:
        """Set new MettaGrid configuration (for curriculum compatibility)."""
        pass

    def get_episode_rewards(self) -> np.ndarray:
        """Get episode rewards (for curriculum compatibility)."""
        return np.array([0.0])

    def close(self) -> None:
        """Clean up environment."""
        pass

    # PufferLib async interface methods
    def async_reset(self, seed: int | None = None) -> np.ndarray:
        """Async reset method for pufferlib compatibility."""
        obs, info = self.reset(seed)
        # Set up results for recv() to use during reset flow
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminals = np.zeros(self.num_agents, dtype=bool)
        truncations = np.zeros(self.num_agents, dtype=bool)
        self._step_results = (obs, rewards, terminals, truncations, info)
        return obs

    def send(self, actions: np.ndarray) -> None:
        """Send actions to environment (pufferlib async interface)."""
        self._step_results = self.step(actions)

    def recv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
        """Receive step results (pufferlib async interface)."""
        if not hasattr(self, "_step_results"):
            raise RuntimeError("Must call send() before recv()")

        obs, rewards, terminals, truncations, info = self._step_results
        info_list = [info.copy() for _ in range(self.num_agents)]
        lives = np.ones(self.num_agents, dtype=np.float32)
        scores = rewards.copy()
        return obs, rewards, terminals, truncations, info_list, lives, scores


# Configuration Classes - same as before but simpler
class TribalGameConfig(Config):
    """Configuration for tribal game mechanics."""

    max_steps: int = Field(default=2000, ge=0)
    ore_per_battery: int = Field(default=3)
    batteries_per_heart: int = Field(default=2)
    enable_combat: bool = Field(default=True)
    clippy_spawn_rate: float = Field(default=1.0, ge=0, le=1)
    clippy_damage: int = Field(default=1)
    heart_reward: float = Field(default=1.0)
    ore_reward: float = Field(default=0.1)
    battery_reward: float = Field(default=0.8)
    survival_penalty: float = Field(default=-0.01)
    death_penalty: float = Field(default=-5.0)

    @property
    def num_agents(self) -> int:
        """Number of agents (compile-time constant)."""
        return MAP_AGENTS


class TribalEnvConfig(Config):
    """Configuration for Nim tribal environments."""

    environment_type: str = "tribal"
    label: str = Field(default="tribal")
    game: TribalGameConfig = Field(default_factory=TribalGameConfig)
    desync_episodes: bool = Field(default=True)
    render_mode: Optional[str] = Field(default=None)

    def get_observation_space(self) -> Dict[str, Any]:
        """Get tribal environment observation space."""
        return {
            "shape": (MAX_TOKENS_PER_AGENT, 3),  # Token format
            "dtype": "uint8",
            "type": "Box",
        }

    def get_action_space(self) -> Dict[str, Any]:
        """Get tribal environment action space."""
        return {
            "shape": (2,),
            "dtype": "uint8",
            "type": "MultiDiscrete",
            "nvec": [6, 8],
        }

    def create_environment(self, **kwargs) -> Any:
        """Create tribal environment instance."""
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
    """Create a tribal environment instance using direct pointer access."""
    return TribalGridEnv(config)


def make_tribal_puffer_env(**config) -> TribalGridEnv:
    """Create a tribal PufferLib environment instance (alias for make_tribal_env)."""
    return TribalGridEnv(config)
