from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch.nn as nn
from pydantic import ConfigDict

from mettagrid.base_config import Config


class ComponentConfig(Config, ABC):
    """Abstract base class for component configurations."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    name: str

    @abstractmethod
    def make_component(self, env: Any = None) -> nn.Module:
        """Create a component instance from configuration."""
        pass
