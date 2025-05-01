from math import prod

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase, ParamLayer


class Linear(ParamLayer):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = [self._nn_params.out_features]
        assert len(self._in_tensor_shapes[0]) == 1, (
            "_input_tensor_shape for Linear should be 1d (ignoring batch dimension)"
        )
        return nn.Linear(self._in_tensor_shapes[0][0], **self._nn_params)


class ReLU(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.ReLU()


class LayerNorm(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.LayerNorm(self._in_tensor_shapes[0][0], **self._nn_params)


class Bilinear(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = [self._nn_params.out_features]

        self._nn_params["in1_features"] = self._in_tensor_shapes[0][0]
        self._nn_params["in2_features"] = self._in_tensor_shapes[1][0]
        self._nn_params = dict(self._nn_params)  # need to convert from omegaconf DictConfig
        return nn.Bilinear(**self._nn_params)

    def _forward(self, td: TensorDict):
        # input_1 = td[self._input_source[0]]
        # input_2 = td[self._input_source[1]]
        input_1 = td[self._sources[0]["name"]]
        input_2 = td[self._sources[1]["name"]]
        td[self._name] = self._net(input_1, input_2)
        return td


class Embedding(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        # output shape [0] is the number of indices used in the forward pass which can change
        # no child layer should be sensitive to this dimension
        self._out_tensor_shape = [0, self._nn_params["embedding_dim"]]

        net = nn.Embedding(**self._nn_params)

        weight_limit = 0.1
        nn.init.orthogonal_(net.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(net.weight))
            net.weight.mul_(weight_limit / max_abs_value)

        return net


class Conv2d(ParamLayer):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._set_conv_dims()
        return nn.Conv2d(self._in_tensor_shapes[0][0], **self._nn_params)

    def _set_conv_dims(self):
        """Calculate flattened width and height. This allows us to change obs width and height."""
        assert len(self._in_tensor_shapes[0]) == 3, "Conv2d input tensor shape must be 3d (ignoring batch dimension)"
        self._input_height = self._in_tensor_shapes[0][1]
        self._input_width = self._in_tensor_shapes[0][2]

        if not hasattr(self._nn_params, "padding") or self._nn_params.padding is None:
            self._nn_params.padding = 0

        self._output_height = (
            (self._input_height + 2 * self._nn_params.padding - self._nn_params.kernel_size) / self._nn_params.stride
        ) + 1
        self._output_width = (
            (self._input_width + 2 * self._nn_params.padding - self._nn_params.kernel_size) / self._nn_params.stride
        ) + 1

        if not self._output_height.is_integer() or not self._output_width.is_integer():
            raise ValueError(f"CNN {self._name} output dimensions must be integers. Adjust padding or kernel size.")

        self._output_height = int(self._output_height)
        self._output_width = int(self._output_width)

        self._out_tensor_shape = [self._nn_params.out_channels, self._output_height, self._output_width]


class MaxPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.MaxPool1d(self._in_tensor_shapes[0][0], **self._nn_params)


class MaxPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.MaxPool2d(self._in_tensor_shapes[0][0], **self._nn_params)


class AdaptiveAvgPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.AdaptiveAvgPool1d(self._in_tensor_shapes[0][0], **self._nn_params)


class AdaptiveAvgPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.AdaptiveAvgPool2d(self._in_tensor_shapes[0][0], **self._nn_params)


class AdaptiveMaxPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.AdaptiveMaxPool1d(self._in_tensor_shapes[0][0], **self._nn_params)


class AdaptiveMaxPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.AdaptiveMaxPool2d(self._in_tensor_shapes[0][0], **self._nn_params)


class AvgPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.AvgPool1d(self._in_tensor_shapes[0][0], **self._nn_params)


class AvgPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.AvgPool2d(self._in_tensor_shapes[0][0], **self._nn_params)


class Dropout(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.Dropout(**self._nn_params)


class Dropout2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.Dropout2d(**self._nn_params)


class AlphaDropout(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.AlphaDropout(**self._nn_params)


class BatchNorm1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.BatchNorm1d(self._in_tensor_shapes[0][0], **self._nn_params)


class BatchNorm2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.BatchNorm2d(self._in_tensor_shapes[0][0], **self._nn_params)


class Flatten(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = [prod(self._in_tensor_shapes[0])]
        return nn.Flatten()


class Identity(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        return nn.Identity()
