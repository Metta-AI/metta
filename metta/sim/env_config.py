"""
Generic Environment Configuration Base Classes.

This module provides a generic interface for different environment types
to work with the Metta training infrastructure.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from metta.mettagrid.config import Config


class EnvConfig(Config, ABC):
    """
    Base class for all environment configurations.
    
    This allows different environment types (MettaGrid, Tribal, etc.)
    to work with the same training infrastructure.
    """
    
    # Common environment parameters
    environment_type: str
    num_agents: int
    max_steps: Optional[int] = None
    
    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        """Get the observation space for this environment."""
        pass
        
    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """Get the action space for this environment."""
        pass
        
    @abstractmethod
    def create_environment(self, **kwargs) -> Any:
        """Create the actual environment instance."""
        pass


class TribalEnvConfig(EnvConfig):
    """Configuration for Nim tribal environments."""
    
    environment_type: str = "tribal"
    render_mode: Optional[str] = None
    bindings_path: Optional[str] = None
    
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
        
        return make_tribal_env(
            num_agents=self.num_agents,
            max_steps=self.max_steps or 2000,
            render_mode=self.render_mode,
            **kwargs
        )