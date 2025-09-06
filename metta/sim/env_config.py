"""
Tribal Environment Configuration.

This module provides configuration for the Nim-based tribal environment.
"""

from typing import Any, Dict, Optional, Literal
from pydantic import Field

from metta.mettagrid.config import Config


class TribalGameConfig(Config):
    """Configuration for tribal game mechanics."""
    
    # Core game parameters
    num_agents: int = Field(default=15, ge=1, description="Number of agents in the environment")
    max_steps: int = Field(default=2000, ge=0, description="Maximum steps per episode (0 = no limit)")
    episode_truncates: bool = Field(default=False, description="Use truncation vs termination")
    
    # Observation space configuration
    obs_width: int = Field(default=24, description="Width of agent observations")
    obs_height: int = Field(default=24, description="Height of agent observations")
    obs_layers: int = Field(default=8, description="Number of observation layers")
    
    # Map configuration
    map_width: int = Field(default=64, description="Total map width")
    map_height: int = Field(default=64, description="Total map height")
    num_villages: int = Field(default=3, description="Number of villages on the map")
    
    # Resource configuration
    resource_spawn_rate: float = Field(default=0.1, ge=0, le=1, description="Rate of resource spawning")
    ore_per_battery: int = Field(default=3, description="Ore required to craft battery")
    batteries_per_heart: int = Field(default=2, description="Batteries required at altar for hearts")
    
    # Combat configuration  
    enable_combat: bool = Field(default=True, description="Enable agent combat")
    clippy_spawn_rate: float = Field(default=0.05, ge=0, le=1, description="Rate of enemy spawning")
    clippy_damage: int = Field(default=1, description="Damage dealt by enemies")
    
    # Reward configuration
    heart_reward: float = Field(default=10.0, description="Reward for creating hearts")
    ore_reward: float = Field(default=0.1, description="Reward for collecting ore")
    battery_reward: float = Field(default=1.0, description="Reward for crafting batteries")
    survival_penalty: float = Field(default=-0.01, description="Per-step survival penalty")
    death_penalty: float = Field(default=-5.0, description="Penalty for agent death")


class TribalEnvConfig(Config):
    """Configuration for Nim tribal environments."""
    
    environment_type: str = "tribal"
    label: str = Field(default="tribal", description="Environment label")
    
    # Game configuration
    game: TribalGameConfig = Field(default_factory=TribalGameConfig)
    
    # Environment settings
    desync_episodes: bool = Field(default=True, description="Desynchronize episode resets")
    num_envs: Optional[int] = Field(default=None, description="Number of parallel environments")
    render_mode: Optional[str] = Field(default=None, description="Rendering mode (human, rgb_array)")
    bindings_path: Optional[str] = Field(default=None, description="Path to Nim bindings")
    
    # Vectorization settings
    batch_size: Optional[int] = Field(default=None, description="Batch size for vectorized steps")
    async_envs: bool = Field(default=False, description="Use async environment stepping")
    num_threads: int = Field(default=1, description="Number of threads for environment")
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Get tribal environment observation space."""
        # Import here to avoid circular dependencies
        try:
            from metta.sim.tribal_genny import OBSERVATION_LAYERS, OBSERVATION_WIDTH, OBSERVATION_HEIGHT
            return {
                "shape": (OBSERVATION_LAYERS, OBSERVATION_HEIGHT, OBSERVATION_WIDTH),
                "dtype": "uint8",
                "type": "Box"
            }
        except ImportError:
            # Fallback values if bindings not available
            return {
                "shape": (8, 24, 24),  # Default tribal obs space
                "dtype": "uint8", 
                "type": "Box"
            }
    
    def get_action_space(self) -> Dict[str, Any]:
        """Get tribal environment action space."""
        return {
            "shape": (2,),  # [action_type, argument]
            "dtype": "int32",
            "type": "MultiDiscrete",
            "nvec": [6, 8]  # 6 action types, 8 directions/targets
        }
    
    def create_environment(self, **kwargs) -> Any:
        """Create tribal environment instance."""
        from metta.sim.tribal_genny import make_tribal_env
        
        # Convert configuration to dictionary format expected by tribal_genny
        config = {
            'num_agents': self.game.num_agents,
            'max_steps': self.game.max_steps,
            'enable_combat': self.game.enable_combat,
            'heart_reward': self.game.heart_reward,
            'battery_reward': self.game.battery_reward,
            'ore_reward': self.game.ore_reward,
            'survival_penalty': self.game.survival_penalty,
            'death_penalty': self.game.death_penalty,
            'resource_spawn_rate': self.game.resource_spawn_rate,
            'clippy_spawn_rate': self.game.clippy_spawn_rate,
            'render_mode': self.render_mode,
            **kwargs
        }
        
        return make_tribal_env(**config)