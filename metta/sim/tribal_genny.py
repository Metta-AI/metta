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
    MAX_TOKENS_PER_AGENT = tribal.MAX_TOKENS_PER_AGENT
    NUM_ACTION_TYPES = tribal.NUM_ACTION_TYPES

    # Helper functions
    default_max_steps = tribal.default_max_steps
    default_tribal_config = tribal.default_tribal_config
    check_error = tribal.check_error
    take_error = tribal.take_error
    get_action_names = tribal.get_action_names
    get_max_action_args = tribal.get_max_action_args
    get_feature_normalizations = tribal.get_feature_normalizations

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

        # Cache dimensions (compile-time constants from Nim)
        self.num_agents = MAP_AGENTS
        self.observation_layers = OBSERVATION_LAYERS
        self.observation_width = OBSERVATION_WIDTH
        self.observation_height = OBSERVATION_HEIGHT

        # Action space info (from Nim)
        self.num_action_types = NUM_ACTION_TYPES
        self.max_argument = 8  # Max argument value (0-7 for directions)

        # Add gym spaces for pufferlib compatibility
        import gymnasium as gym

        # Token observation space: [num_tokens, 3] where each token is [coord_byte, layer, value]
        self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(MAX_TOKENS_PER_AGENT, 3), dtype=np.uint8)

        # Action space: [action_type, argument]
        self.single_action_space = gym.spaces.MultiDiscrete([self.num_action_types, self.max_argument])

        # Cache for compatibility
        self.obs_width = self.observation_width
        self.obs_height = self.observation_height
        
        # Add height and width properties for replay system compatibility
        self.height = MAP_HEIGHT
        self.width = MAP_WIDTH
        
        # Add grid_objects property for MettaScope visualization compatibility
        # Return empty dict since tribal uses a different object system
        self.grid_objects = {}

        # Feature normalizations from Nim
        feature_norms_seq = get_feature_normalizations()
        self.feature_normalizations = {i: feature_norms_seq[i] for i in range(len(feature_norms_seq))}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observations."""
        # Reset the Nim environment (our binding doesn't take seed parameter)
        self._nim_env.reset_env()

        # Get token observations directly from Nim
        obs_data = self._nim_env.get_token_observations()
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
        obs_data = self._nim_env.get_token_observations()
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

    def _convert_observations(self, token_seq: SeqInt) -> np.ndarray:
        """Convert genny token observation data directly to numpy format."""
        token_size = 3  # [coord_byte, layer, value]
        expected_size = self.num_agents * MAX_TOKENS_PER_AGENT * token_size
        
        # Convert SeqInt directly to token format
        if len(token_seq) == expected_size:
            # Convert to python list then numpy and reshape
            token_data = [token_seq[i] for i in range(len(token_seq))]
            token_array = np.array(token_data, dtype=np.uint8)
            token_obs = token_array.reshape((self.num_agents, MAX_TOKENS_PER_AGENT, token_size))
        else:
            # Fallback: create empty tokens and fill what we have
            token_obs = np.full((self.num_agents, MAX_TOKENS_PER_AGENT, token_size), 0xFF, dtype=np.uint8)
            index = 0
            for agent_id in range(self.num_agents):
                for token_idx in range(MAX_TOKENS_PER_AGENT):
                    for dim in range(token_size):
                        if index < len(token_seq):
                            token_obs[agent_id, token_idx, dim] = token_seq[index]
                            index += 1
                        else:
                            break
                    if index >= len(token_seq):
                        break
                if index >= len(token_seq):
                    break
        
        return token_obs


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
        # Tribal environments use compile-time constants, so no dynamic reconfiguration needed
        pass

    def get_episode_rewards(self) -> np.ndarray:
        """Get episode rewards (for curriculum compatibility)."""
        # Return dummy rewards array for compatibility
        return np.array([0.0])

    def close(self) -> None:
        """Clean up environment."""
        # Nim's GC will handle cleanup
        pass

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
        # Return feature spec for each observation layer using Nim data
        features = {}
        feature_norms_seq = get_feature_normalizations()
        for layer_id in range(len(feature_norms_seq)):
            features[f"layer_{layer_id}"] = {
                "id": layer_id,
                "normalization": feature_norms_seq[layer_id],
            }
        return features

    # Pufferlib async interface methods
    def async_reset(self, seed: int | None = None) -> np.ndarray:
        """Async reset method for pufferlib compatibility."""
        obs, info = self.reset(seed)
        
        # Set up results for recv() to use during reset flow
        # Create dummy rewards/terminals/truncations for initial reset
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminals = np.zeros(self.num_agents, dtype=bool)
        truncations = np.zeros(self.num_agents, dtype=bool)
        
        self._step_results = (obs, rewards, terminals, truncations, info)
        return obs

    def send(self, actions: np.ndarray) -> None:
        """Send actions to environment (pufferlib async interface)."""
        # Store the step results for recv() to return
        self._step_results = self.step(actions)

    def recv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
        """Receive step results (pufferlib async interface)."""
        if not hasattr(self, '_step_results'):
            raise RuntimeError("Must call send() before recv()")
        
        obs, rewards, terminals, truncations, info = self._step_results
        
        # Convert info dict to list of dicts (one per agent) 
        info_list = [info.copy() for _ in range(self.num_agents)]
        
        # Add lives and scores arrays (dummy values for tribal env)
        lives = np.ones(self.num_agents, dtype=np.float32)  # All agents alive
        scores = rewards.copy()  # Use rewards as scores
        
        return obs, rewards, terminals, truncations, info_list, lives, scores


# Configuration Classes - Minimal pass-through to Nim
class TribalGameConfig:
    """Minimal pass-through wrapper for Nim TribalGameConfig.
    
    All configuration values come directly from Nim - no Python-side duplication.
    This is just a thin compatibility layer for the existing Python API.
    """
    
    def __init__(self, _nim_config=None, **overrides):
        # Store the Nim config object directly - no Python fields
        self._nim_config = _nim_config or default_tribal_config().game
        
        # Apply any overrides directly to the Nim object
        for key, value in overrides.items():
            if hasattr(self._nim_config, key):
                setattr(self._nim_config, key, value)
            else:
                raise AttributeError(f"Unknown config parameter: {key}")
    
    def __getattr__(self, name: str):
        """Delegate all attribute access to the Nim config object."""
        return getattr(self._nim_config, name)
    
    def __setattr__(self, name: str, value):
        """Delegate all attribute setting to the Nim config object."""
        if name.startswith('_'):
            # Private attributes stay on Python object
            super().__setattr__(name, value)
        else:
            # Everything else goes to Nim
            setattr(self._nim_config, name, value)
    
    @property
    def num_agents(self) -> int:
        """Number of agents (compile-time constant)."""
        return MAP_AGENTS
    
    @classmethod
    def from_nim_defaults(cls) -> "TribalGameConfig":
        """Create config with defaults from Nim environment."""
        return cls()  # That's it! No field-by-field copying needed


class TribalEnvConfig:
    """Minimal pass-through configuration for tribal environments.
    
    Contains only Python-specific concerns (label, render mode).
    All game configuration passes through directly to Nim.
    """
    
    def __init__(self, label: str = "tribal", desync_episodes: bool = True, 
                 render_mode: Optional[str] = None, game: Optional[TribalGameConfig] = None, **game_overrides):
        self.environment_type = "tribal"
        self.label = label
        self.desync_episodes = desync_episodes
        self.render_mode = render_mode
        
        # Game config is just a pass-through to Nim
        if game is not None:
            self.game = game
        else:
            self.game = TribalGameConfig(**game_overrides)
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Provide Pydantic schema for compatibility with TaskGeneratorConfig."""
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls._pydantic_validator,
            core_schema.dict_schema(),
        )
    
    @classmethod  
    def _pydantic_validator(cls, value):
        """Convert dict to TribalEnvConfig instance for Pydantic compatibility."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return value
    
    def model_copy(self, *, deep: bool = False) -> "TribalEnvConfig":
        """Create a copy of this config (for Pydantic compatibility)."""
        # Create a new game config (deep copy the Nim config if needed)
        if deep:
            # Create a new game config with the same values
            new_game = TribalGameConfig()
            for attr_name in ['max_steps', 'ore_per_battery', 'batteries_per_heart', 
                             'enable_combat', 'clippy_spawn_rate', 'clippy_damage',
                             'heart_reward', 'ore_reward', 'battery_reward', 
                             'survival_penalty', 'death_penalty']:
                if hasattr(self.game, attr_name):
                    setattr(new_game, attr_name, getattr(self.game, attr_name))
        else:
            new_game = self.game
            
        # Create new instance with copied values
        return TribalEnvConfig(
            label=self.label,
            desync_episodes=self.desync_episodes,
            render_mode=self.render_mode,
            game=new_game
        )

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
            "nvec": [6, 8],  # 6 action types, 8 max argument value (0-7 for directions)
        }

    @classmethod
    def with_nim_defaults(cls, **overrides) -> "TribalEnvConfig":
        """Create config with defaults from Nim environment."""
        game_config = TribalGameConfig.from_nim_defaults()
        return cls(
            game=game_config,
            label=overrides.get("label", "tribal"),
            desync_episodes=overrides.get("desync_episodes", True),
            render_mode=overrides.get("render_mode"),
        )

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
