"""Simple curriculum store for managing curriculum definitions."""

from typing import Dict, Optional

from metta.mettagrid.curriculum.curriculum_config import CurriculumConfig


class CurriculumStore:
    """Simple store for curriculum configurations."""
    
    _curricula: Dict[str, CurriculumConfig] = {}
    
    @classmethod
    def register(cls, name: str, config: CurriculumConfig) -> None:
        """Register a curriculum configuration."""
        cls._curricula[name] = config
    
    @classmethod
    def get(cls, name: str) -> Optional[CurriculumConfig]:
        """Get a curriculum configuration by name."""
        return cls._curricula.get(name)
    
    @classmethod
    def list(cls) -> list[str]:
        """List all registered curriculum names."""
        return list(cls._curricula.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered curricula (mainly for testing)."""
        cls._curricula.clear()


# Global store instance
curriculum_store = CurriculumStore()