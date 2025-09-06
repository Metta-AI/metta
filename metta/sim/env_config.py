"""
Tribal Environment Configuration.

This module provides configuration for the Nim-based tribal environment.
"""

from typing import Any, Dict, Optional, Literal
from pydantic import Field

from metta.mettagrid.config import Config


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
    clippy_spawn_rate: float = Field(default=0.05, ge=0, le=1, description="Rate of enemy spawning")
    clippy_damage: int = Field(default=1, description="Damage dealt by enemies")
    
    # Reward configuration (exact arena_basic_easy_shaped values)
    heart_reward: float = Field(default=1.0, description="Reward for creating hearts")
    ore_reward: float = Field(default=0.1, description="Reward for collecting ore")
    battery_reward: float = Field(default=0.8, description="Reward for crafting batteries")
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
    render_mode: Optional[str] = Field(default=None, description="Rendering mode (human, rgb_array)")
    
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
            'max_steps': self.game.max_steps,
            'ore_per_battery': self.game.ore_per_battery,
            'batteries_per_heart': self.game.batteries_per_heart,
            'enable_combat': self.game.enable_combat,
            'clippy_spawn_rate': self.game.clippy_spawn_rate,
            'clippy_damage': self.game.clippy_damage,
            'heart_reward': self.game.heart_reward,
            'battery_reward': self.game.battery_reward,
            'ore_reward': self.game.ore_reward,
            'survival_penalty': self.game.survival_penalty,
            'death_penalty': self.game.death_penalty,
            'render_mode': self.render_mode,
            **kwargs
        }
        
        return make_tribal_env(**config)