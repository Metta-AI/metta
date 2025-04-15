import torch.nn as nn

from .metta_layer import LayerBase, ParamLayer

class Linear(ParamLayer):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Linear(self._input_size, self._output_size, **self._nn_params)


class Conv1d(ParamLayer):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Conv1d(self._input_size, self._output_size, **self._nn_params)


class Conv2d(ParamLayer):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Conv2d(self._input_size, self._output_size, **self._nn_params)


class MaxPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.MaxPool1d(self._input_size, **self._nn_params)


class MaxPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.MaxPool2d(self._input_size, **self._nn_params)


class AdaptiveAvgPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveAvgPool1d(self._input_size, **self._nn_params)


class AdaptiveAvgPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveAvgPool2d(self._input_size, **self._nn_params)


class AdaptiveMaxPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveMaxPool1d(self._input_size, **self._nn_params)


class AdaptiveMaxPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AdaptiveMaxPool2d(self._input_size, **self._nn_params)


class AvgPool1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AvgPool1d(self._input_size, **self._nn_params)


class AvgPool2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AvgPool2d(self._input_size, **self._nn_params)


class Dropout(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Dropout(**self._nn_params)


class Dropout2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Dropout2d(**self._nn_params)


class AlphaDropout(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.AlphaDropout(**self._nn_params)


class BatchNorm1d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.BatchNorm1d(self._input_size, **self._nn_params)


class BatchNorm2d(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.BatchNorm2d(self._input_size, **self._nn_params)


class Flatten(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Flatten()


class Identity(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Identity()
