from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

from mettagrid.base_config import Config


class ComponentConfig(Config, ABC):
    """Abstract base class for component configurations."""

    name: str

    @abstractmethod
    def make_component(self, env: Any = None) -> nn.Module:
        """Create a component instance from configuration."""
        pass
