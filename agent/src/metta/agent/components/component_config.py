from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from mettagrid.base_config import Config


class ComponentConfig(Config, ABC):
    """Abstract base class for component configurations."""

    name: str

    @abstractmethod
    def make_component(self, env: Any = None) -> nn.Module:
        """Create a component instance from configuration."""
        pass
