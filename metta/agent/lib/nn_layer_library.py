from math import prod

import torch
import torch.nn as nn
from tensordict import TensorDict
from e3nn import o3

from metta.agent.lib.metta_layer import ParamLayer, LayerBase

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
        self._out_tensor_shape = [self._nn_params.out_features]

        self._nn_params['in1_features'] = self._in_tensor_shape[0][0]
        self._nn_params['in2_features'] = self._in_tensor_shape[1][0]
        self._nn_params = dict(self._nn_params) # need to convert from omegaconf DictConfig
        return nn.Bilinear(
            **self._nn_params
        )
    
    def _forward(self, td: TensorDict):
        input_1 = td[self._input_source[0]]
        input_2 = td[self._input_source[1]]
        td[self._name] = self._net(input_1, input_2)
        return td
    
class BilinearE3nn(LayerBase):
    def __init__(self, **cfg):
        # We need to determine the output shape *before* calling super().__init__
        # if LayerBase uses it to initialize things based on _out_tensor_shape
        # nn_params = cfg.get('nn_params', {})
        # self._determined_out_features = nn_params.get('out_features', 0) # Store it temporarily
        super().__init__(**cfg)
        # Ensure the output shape is correctly set after super init if needed
        # self._out_tensor_shape = [self._determined_out_features]

    def _make_net(self):
        self._out_tensor_shape = [self._nn_params.out_features]
        # Extract original feature counts
        self._nn_params['in1_features'] = self._in_tensor_shape[0][0]
        self._nn_params['in2_features'] = self._in_tensor_shape[1][0]
        self._nn_params = dict(self._nn_params)

        # in1_features = self._in_tensor_shape[0][0]
        # in2_features = self._in_tensor_shape[1][0]
        # out_features = self._determined_out_features # Use the stored value

        # Define Irreps: Treat all features as scalars ('0e')
        irreps_in1 = o3.Irreps(f"{self._nn_params['in1_features']}x0e")
        irreps_in2 = o3.Irreps(f"{self._nn_params['in2_features']}x0e")
        irreps_out = o3.Irreps(f"{self._nn_params['out_features']}x0e")

        # --- Handling Bias ---
        # nn.Bilinear has an optional bias term. e3nn.o3.FullyConnectedTensorProduct doesn't directly.
        # Option 1 (Chosen here): Ignore bias. The performance benefit might outweigh the lack of bias.
        # Option 2: Add bias manually after the layer in _forward.
        # Option 3: Incorporate bias via constant input features (more complex).
        # We'll go with Option 1 for simplicity. Check if 'bias' was in your original config.
        if self._nn_params.get('bias', True): # Default bias is True for nn.Bilinear
             print(f"Warning: Original nn.Bilinear likely had bias=True. "
                   f"e3nn.o3.FullyConnectedTensorProduct does not have a direct bias parameter. "
                   f"Bias term is currently omitted for layer '{self._name}'.")

        # Create the e3nn layer
        # You might want to expose other e3nn params (shared_weights, internal_weights etc.) via your config
        return o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out
            # Add other e3nn parameters here if needed, e.g.,
            # shared_weights=True, # Default
            # internal_weights=None # Default
        )

    def _forward(self, td: dict): # Using dict instead of TensorDict for simplicity here
        # Ensure inputs are standard tensors (e3nn handles the Irreps internally)
        input_1 = td[self._input_source[0]]
        input_2 = td[self._input_source[1]]

        # Ensure inputs are 2D: [N, features]
        if input_1.ndim != 2 or input_2.ndim != 2:
            raise ValueError(f"Inputs must be 2D [N, features]. Got shapes: {input_1.shape}, {input_2.shape}")

        # Apply the e3nn layer
        output = self._net(input_1, input_2)
        td[self._name] = output
        return td

class Embedding(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        net = nn.Embedding(
            **self._nn_params
        )
        if self.initialization.lower() == 'orthogonal':
            weight_limit = 0.1
            nn.init.orthogonal_(net.weight)
            with torch.no_grad():
                max_abs_value = torch.max(torch.abs(net.weight))
                net.weight.mul_(weight_limit / max_abs_value)
        elif self.initialization.lower() == 'max_1':
            nn.init.uniform_(net.weight, a=-1, b=1)
        elif self.initialization.lower() == 'max_0_1':
            nn.init.uniform_(net.weight, a=-1, b=1)
        elif self.initialization.lower() == 'max_0_01':
            nn.init.uniform_(net.weight, a=-0.01, b=0.01)
        return net

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
        self._out_tensor_shape = [prod(self._in_tensor_shape)]
        return nn.Flatten()

class Identity(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        return nn.Identity()

