import torch.nn as nn

from .metta_layer import ParamLayer, LayerBase

class Linear(ParamLayer):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = [self._nn_params.out_features]
        return nn.Linear(
            self._in_tensor_shape[0],
            **self._nn_params
        )
    
class Bilinear(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._nn_params['in1_features'] = self._in_tensor_shape[0]
        self._nn_params['in2_features'] = self._in_tensor_shape[1]
        self._out_tensor_shape = [self._nn_params.out_features]
        return nn.Bilinear(
            **self._nn_params
        )
    
class Embedding(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = [self._nn_params.embedding_dim]
        return nn.Embedding(
            **self._nn_params
        )

class Conv2d(ParamLayer):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._set_conv_dims()
        return nn.Conv2d(
            self._in_tensor_shape[0],
            **self._nn_params
        )
    
    def _set_conv_dims(self):
        ''' Calculate flattened width and height. This allows us to change obs width and height.'''
        self._input_height = self._in_tensor_shape[1]
        self._input_width = self._in_tensor_shape[2]

        if not hasattr(self._nn_params, 'padding') or self._nn_params.padding is None:
            self._nn_params.padding = 0

        self._output_height = ((self._input_height + 2 * self._nn_params.padding - self._nn_params.kernel_size) / self._nn_params.stride) + 1
        self._output_width = ((self._input_width + 2 * self._nn_params.padding - self._nn_params.kernel_size) / self._nn_params.stride) + 1

        if not self._output_height.is_integer() or not self._output_width.is_integer():
            raise ValueError(f"CNN {self._name} output dimensions must be integers. Adjust padding or kernel size.")
        
        self._output_height = int(self._output_height)
        self._output_width = int(self._output_width)

        self._out_tensor_shape = [self._nn_params.out_channels, self._output_height, self._output_width]

class MaxPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.MaxPool1d(
            self._in_tensor_shape[0],
            **self._nn_params
        )

class MaxPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.MaxPool2d(
            self._in_tensor_shape[0],
            **self._nn_params
        )

class AdaptiveAvgPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveAvgPool1d(
            self._in_tensor_shape[0],
            **self._nn_params
        )

class AdaptiveAvgPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveAvgPool2d(
            self._in_tensor_shape[0],
            **self._nn_params
        )

class AdaptiveMaxPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveMaxPool1d(
            self._in_tensor_shape[0],
            **self._nn_params
        )

class AdaptiveMaxPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveMaxPool2d(
            self._in_tensor_shape[0],
            **self._nn_params
        )

class AvgPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AvgPool1d(
            self._in_tensor_shape[0],
            **self._nn_params
        )

class AvgPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AvgPool2d(
            self._in_tensor_shape[0],
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
            self._in_tensor_shape[0],
            **self._nn_params
        )

class BatchNorm2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.BatchNorm2d(
            self._in_tensor_shape[0],
            **self._nn_params
        )

class Flatten(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = [sum(self._in_tensor_shape)]
        return nn.Flatten()

class Identity(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Identity()

