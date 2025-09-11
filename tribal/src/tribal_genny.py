"""
Simplified Tribal Environment - Direct pass-through to Nim with pointer-based interface.
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
    Path(__file__).parent.parent / "bindings" / "generated",
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

    # Extract classes and functions we need
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

    # Functions - direct pass-through to Nim
    get_action_names = tribal.get_action_names
    get_max_action_args = tribal.get_max_action_args
    get_feature_normalizations = tribal.get_feature_normalizations
    is_emulated = tribal.is_emulated
    is_done = tribal.is_done

    # Pointer-based functions
    reset_and_get_obs_pointer = tribal.reset_and_get_obs_pointer
    step_with_pointers = tribal.step_with_pointers

except ImportError as e:
    raise ImportError(
        f"Could not import tribal bindings: {e}\nRun 'cd tribal && ./build_bindings.sh' to generate bindings."
    ) from e


class TribalGridEnv:
    """
    Minimal pass-through wrapper for Nim tribal environment.

    Uses direct pointer access for zero-copy performance.
    Python handles memory allocation, Nim reads/writes directly.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None, buf: Optional[Any] = None
    ):
        """Initialize tribal environment."""
        # Set environment variable to signal Python training mode
        os.environ["TRIBAL_PYTHON_CONTROL"] = "1"

        # Create Nim configuration with overrides
        nim_config = default_tribal_config()
        if config:
            for key, value in config.items():
                if hasattr(nim_config.game, key):
                    setattr(nim_config.game, key, value)

        # Create Nim environment instance
        self._nim_env = TribalEnv(nim_config)
        self._config = nim_config

        # Pre-allocate numpy arrays for zero-copy communication
        self.observations = np.zeros((MAP_AGENTS, MAX_TOKENS_PER_AGENT, 3), dtype=np.uint8)
        self.rewards = np.zeros(MAP_AGENTS, dtype=np.float32)
        self.terminals = np.zeros(MAP_AGENTS, dtype=bool)
        self.truncations = np.zeros(MAP_AGENTS, dtype=bool)

        # Environment constants
        self.num_agents = MAP_AGENTS
        self.max_steps = nim_config.game.max_steps
        self.height = MAP_HEIGHT
        self.width = MAP_WIDTH
        self.render_mode = render_mode

        # Gym spaces for compatibility
        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(MAX_TOKENS_PER_AGENT, 3), dtype=np.uint8)
        self.single_action_space = gym.spaces.MultiDiscrete([NUM_ACTION_TYPES, 8])

        # Additional compatibility properties
        self.obs_width = OBSERVATION_WIDTH
        self.obs_height = OBSERVATION_HEIGHT
        self.grid_objects = {}
        feature_norms = get_feature_normalizations()
        self.feature_normalizations = {i: feature_norms[i] for i in range(len(feature_norms))}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment using direct pointer access."""
        # Get pointer to observations array
        obs_ptr_int = self.observations.ctypes.data_as(ctypes.c_void_p).value or 0

        # Reset and get observations written directly to our array
        success = reset_and_get_obs_pointer(self._nim_env, obs_ptr_int)
        if not success:
            raise RuntimeError("Environment reset failed")

        info = {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self._config.game.max_steps,
        }
        return self.observations, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step environment using direct pointer access."""
        if actions.shape != (self.num_agents, 2):
            raise ValueError(f"Actions must have shape ({self.num_agents}, 2), got {actions.shape}")

        # Convert to uint8 and get pointers
        actions_uint8 = actions.astype(np.uint8)
        actions_ptr = actions_uint8.ctypes.data_as(ctypes.c_void_p).value or 0
        obs_ptr = self.observations.ctypes.data_as(ctypes.c_void_p).value or 0
        rewards_ptr = self.rewards.ctypes.data_as(ctypes.c_void_p).value or 0
        terminals_ptr = self.terminals.ctypes.data_as(ctypes.c_void_p).value or 0
        truncations_ptr = self.truncations.ctypes.data_as(ctypes.c_void_p).value or 0

        # Step environment with direct pointer access
        success = step_with_pointers(self._nim_env, actions_ptr, obs_ptr, rewards_ptr, terminals_ptr, truncations_ptr)
        if not success:
            raise RuntimeError("Environment step failed")

        info = {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self._config.game.max_steps,
            "episode_done": self._nim_env.is_episode_done(),
        }
        return self.observations, self.rewards, self.terminals, self.truncations, info

    # Simple pass-through properties and methods
    @property
    def emulated(self) -> bool:
        return is_emulated()

    @property
    def done(self) -> bool:
        return is_done(self._nim_env)

    @property
    def current_step(self) -> int:
        return self._nim_env.get_current_step()

    @property
    def action_names(self) -> list[str]:
        names_seq = get_action_names()
        return [names_seq[i] for i in range(len(names_seq))]

    @property
    def max_action_args(self) -> list[int]:
        args_seq = get_max_action_args()
        return [args_seq[i] for i in range(len(args_seq))]

    def get_observation_features(self) -> dict[str, dict]:
        features = {}
        feature_norms = get_feature_normalizations()
        for layer_id in range(len(feature_norms)):
            features[f"layer_{layer_id}"] = {
                "id": layer_id,
                "normalization": feature_norms[layer_id],
            }
        return features

    def render(self, mode: str = "human") -> Optional[str]:
        if mode == "human":
            return self._nim_env.render_text()
        return None

    def get_episode_stats(self) -> Dict[str, Any]:
        return {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self._config.game.max_steps,
            "episode_done": self._nim_env.is_episode_done(),
        }

    # Minimal compatibility methods
    def set_mg_config(self, config) -> None:
        pass

    def get_episode_rewards(self) -> np.ndarray:
        return np.array([0.0])

    def close(self) -> None:
        pass

    # PufferLib async interface methods
    def async_reset(self, seed: int | None = None) -> np.ndarray:
        obs, info = self.reset(seed)
        # Set up results for recv()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminals = np.zeros(self.num_agents, dtype=bool)
        truncations = np.zeros(self.num_agents, dtype=bool)
        self._step_results = (obs, rewards, terminals, truncations, info)
        return obs

    def send(self, actions: np.ndarray) -> None:
        self._step_results = self.step(actions)

    def recv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
        if not hasattr(self, "_step_results"):
            raise RuntimeError("Must call send() before recv()")

        obs, rewards, terminals, truncations, info = self._step_results
        info_list = [info.copy() for _ in range(self.num_agents)]
        lives = np.ones(self.num_agents, dtype=np.float32)
        scores = rewards.copy()
        return obs, rewards, terminals, truncations, info_list, lives, scores


# Configuration Classes - same interface, minimal implementation
class TribalGameConfig(Config):
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
        return MAP_AGENTS


class TribalEnvConfig(Config):
    environment_type: str = "tribal"
    label: str = Field(default="tribal")
    game: TribalGameConfig = Field(default_factory=TribalGameConfig)
    desync_episodes: bool = Field(default=True)
    render_mode: Optional[str] = Field(default=None)

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "shape": (MAX_TOKENS_PER_AGENT, 3),
            "dtype": "uint8",
            "type": "Box",
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "shape": (2,),
            "dtype": "uint8",
            "type": "MultiDiscrete",
            "nvec": [6, 8],
        }

    def create_environment(self, **kwargs) -> Any:
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
    """Create a tribal environment instance."""
    return TribalGridEnv(config)


def make_tribal_puffer_env(**config) -> TribalGridEnv:
    """Create a tribal PufferLib environment instance."""
    return TribalGridEnv(config)
