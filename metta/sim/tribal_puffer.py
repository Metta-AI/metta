"""
Tribal Environment PufferLib wrapper.

This provides a PufferLib-compatible wrapper for the Tribal environment using
genny-generated bindings. This allows tribal environments to be used directly
with PufferLib training infrastructure while maintaining high performance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from pufferlib import PufferEnv
from typing_extensions import override

# Add the genny-generated bindings to Python path
_BINDINGS_PATHS = [
    Path(__file__).parent.parent.parent / "tribal" / "bindings" / "generated",
]

for path in _BINDINGS_PATHS:
    if path.exists():
        sys.path.insert(0, str(path))
        break

try:
    # Import genny-generated bindings
    import tribal

    # Extract classes and functions
    TribalEnv = tribal.TribalEnv
    SeqInt = tribal.SeqInt

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
    default_tribal_config = tribal.default_tribal_config

except ImportError as e:
    raise ImportError(
        f"Could not import tribal bindings: {e}\nRun the tribal bindings build script to generate bindings."
    ) from e


class TribalPufferEnv(PufferEnv):
    """
    PufferLib wrapper for Tribal Environment.
    
    This class provides full PufferLib compatibility for the Tribal environment
    by inheriting from PufferEnv and implementing the required interface methods.
    It uses the high-performance genny-generated bindings underneath.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        buf: Optional[Any] = None,
    ):
        """
        Initialize PufferLib tribal environment.

        Args:
            config: Configuration dictionary for the environment
            render_mode: Rendering mode (not used for tribal, but kept for compatibility)
            buf: PufferLib buffer object
        """
        # Create Nim configuration object
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

        # Create the Nim environment
        self._nim_env = TribalEnv(nim_config)
        self._config = nim_config
        
        # Environment properties
        self.num_agents = MAP_AGENTS
        self.render_mode = render_mode
        
        # Add compatibility properties expected by the training system
        self.obs_width = OBSERVATION_WIDTH
        self.obs_height = OBSERVATION_HEIGHT
        self.height = MAP_HEIGHT
        self.width = MAP_WIDTH
        
        # Get feature normalizations from Nim
        get_feature_normalizations = tribal.get_feature_normalizations
        feature_norms_seq = get_feature_normalizations()
        self.feature_normalizations = {i: feature_norms_seq[i] for i in range(len(feature_norms_seq))}
        
        # Get action information from Nim
        get_action_names = tribal.get_action_names
        get_max_action_args = tribal.get_max_action_args
        
        names_seq = get_action_names()
        self.action_names = [names_seq[i] for i in range(len(names_seq))]
        
        args_seq = get_max_action_args()
        self.max_action_args = [args_seq[i] for i in range(len(args_seq))]
        
        # Define observation and action spaces
        self._observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(MAX_TOKENS_PER_AGENT, 3), 
            dtype=np.uint8
        )
        self._action_space = gym.spaces.MultiDiscrete([NUM_ACTION_TYPES, 8])

        # Initialize PufferEnv
        PufferEnv.__init__(self, buf=buf)
        
        # Auto-Reset flag
        self._should_reset = False
        
        # Buffers for step results
        self.observations: np.ndarray
        self.terminals: np.ndarray  
        self.truncations: np.ndarray
        self.rewards: np.ndarray

    # PufferLib required properties
    @property
    def single_observation_space(self) -> gym.Space:
        """Single agent observation space for PufferLib."""
        return self._observation_space

    @property
    def single_action_space(self) -> gym.Space:
        """Single agent action space for PufferLib."""
        return self._action_space

    @property
    def emulated(self) -> bool:
        """Native envs do not use emulation (PufferLib compatibility)."""
        return False

    @property
    @override
    def done(self) -> bool:
        """Check if environment is done."""
        return self._should_reset

    def _get_initial_observations(self) -> np.ndarray:
        """Get initial observations after reset."""
        observations, _ = self.reset()
        return observations

    @override
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observations."""
        self._should_reset = False
        
        # Reset the Nim environment
        self._nim_env.reset_env()
        
        # Get token observations
        obs_data = self._nim_env.get_token_observations()
        observations = self._convert_observations(obs_data)
        
        info = {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self._config.game.max_steps,
        }
        
        return observations, info

    @override
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step environment with actions."""
        # Validate input shape
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
            self._should_reset = True
        
        # Check if all agents are done
        if terminals.all() or truncations.all():
            self._should_reset = True

        info = {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self._config.game.max_steps,
            "episode_done": self._nim_env.is_episode_done(),
        }

        return observations, rewards, terminals, truncations, info

    def _convert_observations(self, token_seq: SeqInt) -> np.ndarray:
        """Convert genny token observation data to numpy format."""
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

    def close(self) -> None:
        """Clean up environment."""
        # Nim's GC will handle cleanup
        pass
    
    # Curriculum system compatibility methods
    def set_mg_config(self, config) -> None:
        """Set new MettaGrid configuration (for curriculum compatibility)."""
        # Tribal environments use compile-time constants, so no dynamic reconfiguration needed
        pass

    def get_episode_rewards(self) -> np.ndarray:
        """Get episode rewards (for curriculum compatibility)."""
        # Return dummy rewards array for compatibility
        return np.array([0.0])

    # Additional PufferLib async interface methods for compatibility
    def async_reset(self, seed: Optional[int] = None) -> np.ndarray:
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

    def recv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
        """Receive step results (pufferlib async interface)."""
        if not hasattr(self, "_step_results"):
            raise RuntimeError("Must call send() before recv()")
            
        obs, rewards, terminals, truncations, info = self._step_results
        
        # Convert info dict to list of dicts (one per agent)
        info_list = [info.copy() for _ in range(self.num_agents)]
        
        # Add lives and scores arrays (dummy values for tribal env)
        lives = np.ones(self.num_agents, dtype=np.float32)
        scores = rewards.copy()
        
        return obs, rewards, terminals, truncations, info_list, lives, scores

    def get_observation_features(self) -> dict[str, dict]:
        """Build the features dictionary for initialize_to_environment."""
        # Return feature spec for each observation layer using Nim data
        features = {}
        get_feature_normalizations = tribal.get_feature_normalizations
        feature_norms_seq = get_feature_normalizations()
        for layer_id in range(len(feature_norms_seq)):
            features[f"layer_{layer_id}"] = {
                "id": layer_id,
                "normalization": feature_norms_seq[layer_id],
            }
        return features


def make_tribal_puffer_env(**config) -> TribalPufferEnv:
    """
    Create a tribal PufferLib environment instance.

    Args:
        **config: Configuration parameters for the environment

    Returns:
        TribalPufferEnv instance
    """
    return TribalPufferEnv(config)