import torch.nn as nn

from .metta_layer import ParamLayer, LayerBase

class Linear(ParamLayer):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Linear(
            self._input_size,
            self._output_size,
            **self._nn_params
        )

class Conv2d(ParamLayer):
    def __init__(self, obs_width, obs_height, **cfg):
        super().__init__(**cfg)
        self.obs_width = obs_width
        self.obs_height = obs_height

    def _make_net(self):
        self._calculate_output_dimensions()
        return nn.Conv2d(
            self._input_size,
            self._output_size,
            **self._nn_params
        )
    
    def _calculate_output_dimensions(self):
        ''' Calculate flattened width and height. This allows us to change obs width and height. '''
        if hasattr(self._input_source_component, '_output_width'):
            self._input_width = self._input_source_component._output_width
            self._input_height = self._input_source_component._output_height
        else:
            self._input_width = self.obs_width
            self._input_height = self.obs_height

        if not hasattr(self._nn_params, 'padding') or self._nn_params.padding is None:
            self._nn_params.padding = 0

        self._output_height = ((self._input_height + 2 * self._nn_params.padding - self._nn_params.kernel_size) / self._nn_params.stride) + 1
        self._output_width = ((self._input_width + 2 * self._nn_params.padding - self._nn_params.kernel_size) / self._nn_params.stride) + 1

        if not self._output_height.is_integer() or not self._output_width.is_integer():
            raise ValueError(f"CNN {self._name} output dimensions must be integers. Adjust padding or kernel size.")
        
        self._output_height = int(self._output_height)
        self._output_width = int(self._output_width)
        
        self._flattened_size = self._output_size * self._output_height * self._output_width

class MaxPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.MaxPool1d(
            self._input_size,
            **self._nn_params
        )

class MaxPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.MaxPool2d(
            self._input_size,
            **self._nn_params
        )

class AdaptiveAvgPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveAvgPool1d(
            self._input_size,
            **self._nn_params
        )

class AdaptiveAvgPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveAvgPool2d(
            self._input_size,
            **self._nn_params
        )

class AdaptiveMaxPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveMaxPool1d(
            self._input_size,
            **self._nn_params
        )

class AdaptiveMaxPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveMaxPool2d(
            self._input_size,
            **self._nn_params
        )

class AvgPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AvgPool1d(
            self._input_size,
            **self._nn_params
        )

class AvgPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AvgPool2d(
            self._input_size,
            **self._nn_params
        )

class Dropout(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Dropout(
            **self._nn_params
        )

class Dropout2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Dropout2d(
            **self._nn_params
        )

class AlphaDropout(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AlphaDropout(
            **self._nn_params
        )

class BatchNorm1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.BatchNorm1d(
            self._input_size,
            **self._nn_params
        )

class BatchNorm2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.BatchNorm2d(
            self._input_size,
            **self._nn_params
        )

class Flatten(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        if isinstance(self._input_source_component, Conv2d):
            self._output_size = self._input_source_component._flattened_size
        return nn.Flatten()

class Identity(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Identity()

