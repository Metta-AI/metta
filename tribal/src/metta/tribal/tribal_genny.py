"""
Clean Tribal Environment - Auto-building with smart delegation.

This version automatically rebuilds bindings if needed and uses delegation
patterns instead of verbose manual pass-through methods.
"""

import ctypes
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import Field

from metta.mettagrid.config import Config

# Find tribal root directory - assume we're running from metta root
_metta_root = Path.cwd()
while _metta_root.name != "metta" and _metta_root.parent != _metta_root:
    _metta_root = _metta_root.parent
_tribal_dir = _metta_root / "tribal"


def _ensure_bindings_available():
    """Auto-build tribal bindings if they don't exist or are stale."""
    bindings_dir = _tribal_dir / "bindings" / "generated"
    library_files = list(bindings_dir.glob("libtribal.*"))
    python_binding = bindings_dir / "tribal.py"

    # Check if bindings exist
    if not library_files or not python_binding.exists():
        _build_bindings(_tribal_dir)

    # Add to Python path
    sys.path.insert(0, str(bindings_dir))


def _build_bindings(tribal_dir: Path):
    """Build the tribal bindings using the build script."""
    build_script = _tribal_dir / "build_bindings.sh"
    if not build_script.exists():
        raise RuntimeError(f"Build script not found: {build_script}")

    result = subprocess.run(
        ["bash", str(build_script)],
        cwd=_tribal_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build bindings: {result.stderr}")


# Auto-ensure bindings are available
_ensure_bindings_available()

# Use the same import pattern as the original but simplified
try:
    bindings_file = _tribal_dir / "bindings" / "generated" / "tribal.py"
    spec = importlib.util.spec_from_file_location("tribal", bindings_file)
    tribal = importlib.util.module_from_spec(spec)
    sys.modules["tribal"] = tribal  # Ensure the module is in sys.modules before loading
    spec.loader.exec_module(tribal)

    # Extract key items at import time (like original)
    default_tribal_config = tribal.default_tribal_config
    TribalEnv = tribal.TribalEnv
    MAP_AGENTS = tribal.MAP_AGENTS
    MAX_TOKENS_PER_AGENT = tribal.MAX_TOKENS_PER_AGENT
    MAP_HEIGHT = tribal.MAP_HEIGHT
    MAP_WIDTH = tribal.MAP_WIDTH
    NUM_ACTION_TYPES = tribal.NUM_ACTION_TYPES
    OBSERVATION_WIDTH = tribal.OBSERVATION_WIDTH
    OBSERVATION_HEIGHT = tribal.OBSERVATION_HEIGHT

    # Core functions
    is_emulated = tribal.is_emulated
    is_done = tribal.is_done
    reset_and_get_obs_pointer = tribal.reset_and_get_obs_pointer
    step_with_pointers = tribal.step_with_pointers

except ImportError as e:
    raise ImportError(
        f"Could not import tribal bindings: {e}\nRun 'cd tribal && ./build_bindings.sh' to generate bindings."
    ) from e


class TribalGridEnv:
    """
    Clean tribal environment wrapper with auto-building and smart delegation.

    Uses direct pointer access for zero-copy performance with minimal boilerplate.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None, buf: Optional[Any] = None
    ):
        """Initialize tribal environment."""
        # Set environment variable to signal Python training mode
        os.environ["TRIBAL_PYTHON_CONTROL"] = "1"

        # Create Nim configuration with simple overrides
        nim_config = default_tribal_config()
        if config:
            # Simple dict-based configuration
            config_dict = config if isinstance(config, dict) else getattr(config, "game", config).__dict__
            for key, value in config_dict.items():
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

        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(MAX_TOKENS_PER_AGENT, 3), dtype=np.uint8)
        self.single_action_space = gym.spaces.MultiDiscrete([NUM_ACTION_TYPES, 8])
        self.obs_width = OBSERVATION_WIDTH
        self.obs_height = OBSERVATION_HEIGHT
        self.grid_objects = {}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment using direct pointer access."""
        obs_ptr_int = self.observations.ctypes.data_as(ctypes.c_void_p).value or 0

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

    # Essential properties for RL frameworks
    @property
    def emulated(self) -> bool:
        return False  # Native environment, not emulated

    @property
    def done(self) -> bool:
        return is_done(self._nim_env)

    @property
    def current_step(self) -> int:
        return self._nim_env.get_current_step()

    @property
    def action_names(self) -> list[str]:
        return ["NOOP", "MOVE", "ATTACK", "GET", "SWAP", "PUT"]

    @property
    def max_action_args(self) -> list[int]:
        return [0, 7, 7, 7, 1, 7]

    # PufferLib async interface methods
    def async_reset(self, seed: int | None = None) -> np.ndarray:
        obs, info = self.reset(seed)
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminals = np.zeros(self.num_agents, dtype=bool)
        truncations = np.zeros(self.num_agents, dtype=bool)
        self._step_results = (obs, rewards, terminals, truncations, info)
        return obs

    def send(self, actions: np.ndarray) -> None:
        self._step_results = self.step(actions)

    def recv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Dict], np.ndarray, np.ndarray]:
        if not hasattr(self, "_step_results"):
            raise RuntimeError("Must call send() before recv()")

        obs, rewards, terminals, truncations, info = self._step_results
        info_list = [info.copy() for _ in range(self.num_agents)]
        lives = np.ones(self.num_agents, dtype=np.float32)
        scores = rewards.copy()
        return obs, rewards, terminals, truncations, info_list, lives, scores


# Configuration Classes
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

    def create_environment(self, **kwargs) -> TribalGridEnv:
        return TribalGridEnv({**self.game.__dict__, "render_mode": self.render_mode, **kwargs})


def make_tribal_env(**config) -> TribalGridEnv:
    """Create a tribal environment instance."""
    return TribalGridEnv(config)
