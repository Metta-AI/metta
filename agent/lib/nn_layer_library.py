import torch.nn as nn

from .metta_layer import ParamLayer, LayerBase

class Linear(ParamLayer):
    def __init__(self, agent_attributes, **cfg):
        super().__init__(agent_attributes, **cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.Linear(
            self.input_size,
            self.output_size,
            **nn_params
        )

class Conv1d(ParamLayer):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(agent_attributes, **cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.Conv1d(
            self.input_size,
            self.output_size,
            **nn_params
        )

class Conv2d(ParamLayer):
    def __init__(self, agent_attributes, **cfg):
        super().__init__(agent_attributes, **cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.Conv2d(
            self.input_size,
            self.output_size,
            **nn_params
        )

class MaxPool1d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.MaxPool1d(
            self.input_size,
            **nn_params
        )

class MaxPool2d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.MaxPool2d(
            self.input_size,
            **nn_params
        )

class AdaptiveAvgPool1d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.AdaptiveAvgPool1d(
            self.input_size,
            **nn_params
        )

class AdaptiveAvgPool2d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.AdaptiveAvgPool2d(
            self.input_size,
            **nn_params
        )

class AdaptiveMaxPool1d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.AdaptiveMaxPool1d(
            self.input_size,
            **nn_params
        )

class AdaptiveMaxPool2d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.AdaptiveMaxPool2d(
            self.input_size,
            **nn_params
        )
        
class AvgPool1d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.AvgPool1d(
            self.input_size,
            **nn_params
        )

class AvgPool2d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)

    def _make_layer(self, nn_params={}, **cfg):
        return nn.AvgPool2d(
            self.input_size,
            **nn_params
        )

class Dropout(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)
        
    def _make_layer(self, nn_params={'p': 0.5}, **cfg):
        return nn.Dropout(
            **nn_params
        )
    
class Dropout2d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)
        
    def _make_layer(self, nn_params={'p': 0.5}, **cfg):
        return nn.Dropout2d(
            **nn_params
        )
    
class AlphaDropout(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)
        
    def _make_layer(self, nn_params={'p': 0.5}, **cfg):
        return nn.AlphaDropout(
            **nn_params
        )

class BatchNorm1d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)
        
    def _make_layer(self, nn_params={}, **cfg):
        return nn.BatchNorm1d(
            self.input_size,
            **nn_params
        )

class BatchNorm2d(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        self.cfg = cfg
        super().__init__(**cfg)
        
    def _make_layer(self, nn_params={}, **cfg):
        return nn.BatchNorm2d(
            self.input_size,
            **nn_params
        )

class Flatten(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        super().__init__(**cfg)
        
    def _make_layer(self): # does this need **cfg?
        return nn.Flatten()

class Identity(LayerBase):
    def __init__(self, agent_attributes, **cfg):
        super().__init__(**cfg)
        
    def _make_layer(self): # does this need **cfg?
        return nn.Identity()
        
