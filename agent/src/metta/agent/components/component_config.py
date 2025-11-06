import abc
import typing

import torch.nn as nn

import mettagrid.base_config


class ComponentConfig(mettagrid.base_config.Config, abc.ABC):
    """Abstract base class for component configurations."""

    name: str

    @abc.abstractmethod
    def make_component(self, env: typing.Any = None) -> nn.Module:
        """Create a component instance from configuration."""
        pass
