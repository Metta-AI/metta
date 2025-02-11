import torch.nn as nn
from .metta_layer import ParameterizedLayer, LayerBase

class Linear(ParameterizedLayer):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.Linear(
            self.input_size,
            self.output_size,
            **self.cfg.get('nn_params', {})
        )
        self._parameter_layer_helper()
        self._initialize_weights()

class Conv1d(ParameterizedLayer):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.Conv1d(
            self.input_size,
            self.output_size,
            **self.cfg.get('nn_params', {})
        )
        self._parameter_layer_helper()
        self._initialize_weights()

class Conv2d(ParameterizedLayer):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.Conv2d(
            self.input_size,
            self.output_size,
            **self.cfg.get('nn_params', {})
        )
        self._parameter_layer_helper()
        self._initialize_weights()

class MaxPool1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.MaxPool1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class MaxPool2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.MaxPool2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AdaptiveAvgPool1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AdaptiveAvgPool1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AdaptiveAvgPool2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AdaptiveAvgPool2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AdaptiveMaxPool1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AdaptiveMaxPool1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AdaptiveMaxPool2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AdaptiveMaxPool2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )
        
class AvgPool1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AvgPool1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class AvgPool2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)

    def _initialize_layer(self):
        self.layer = nn.AvgPool2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class Dropout(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.Dropout(
            **self.cfg.get('nn_params', {'p': 0.5})
        )
    
class Dropout2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.Dropout2d(
            **self.cfg.get('nn_params', {'p': 0.5})
        )
    
class AlphaDropout(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.AlphaDropout(
            **self.cfg.get('nn_params', {'p': 0.5})
        )

class BatchNorm1d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.BatchNorm1d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class BatchNorm2d(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
            self.layer = nn.BatchNorm2d(
            self.input_size,
            **self.cfg.get('nn_params', {})
        )

class Flatten(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.Flatten()

class Identity(LayerBase):
    def __init__(self, metta_agent, **cfg):
        self.cfg = cfg
        super().__init__(metta_agent, **cfg)
        
    def _initialize_layer(self):
        self.layer = nn.Identity()
        
