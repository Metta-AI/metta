from math import prod

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase, NNParams, ParamLayer


class Linear(ParamLayer):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        assert self._nn_params.out_features is not None
        self._out_tensor_shape = torch.Size([self._nn_params.out_features])
        assert len(self._in_tensor_shapes[0]) == 1, (
            "_input_tensor_shape for Linear should be 1d (we don't include the batch dimension)"
        )
        return nn.Linear(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class ReLU(LayerBase):
    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.ReLU()


class Tanh(LayerBase):
    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.Tanh()


class LayerNorm(LayerBase):
    """
    Applies Layer Normalization over a mini-batch of inputs

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.LayerNorm(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class Bilinear(LayerBase):
    """
    Applies a bilinear transformation to the incoming data: y = x1 * A * x2^T + b

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        assert self._nn_params.out_features is not None
        assert len(self._in_tensor_shapes) == 2
        assert len(self._in_tensor_shapes[0]) == 1
        assert len(self._in_tensor_shapes[1]) == 1

        self._out_tensor_shape = torch.Size([self._nn_params.out_features])

        return nn.Bilinear(self._in_tensor_shapes[0][0], self._in_tensor_shapes[1][0], self._nn_params.out_features)

    def _forward(self, td: TensorDict):
        assert self._net is not None
        input_1 = td[self.src_component_name(0)]
        input_2 = td[self.src_component_name(1)]
        td[self._name] = self._net(input_1, input_2)
        return td


class ResidualBlock(nn.Module):
    """
    This class cannot be used as a layer in MettaAgent. It is a helper class for ResNetMLP.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Swish(nn.Module):
    """
    This class cannot be used as a layer in MettaAgent. It is a helper class for ResNetMLP.
    """

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class ResNetMLP(LayerBase):
    """
    Applies a residual dense connection to the incoming data: y = x + F(x)
    Input and output shapes are the same. To scale, you need to create an initial and/or final linear layer separately
    that maps to the hidden size you want.
    """

    def __init__(self, depth, **cfg):
        super().__init__(**cfg)
        self._depth = depth

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]

        if self._depth % 4 != 0:
            raise ValueError("Depth must be a multiple of 4.")
        self._num_blocks = self._depth // 4

        hidden_size = self._in_tensor_shapes[0][0]

        self.residual_layers = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(self._num_blocks)])

    def _forward(self, td: TensorDict):
        x = td[self.src_component_name(0)]
        x = self.residual_layers(x)
        td[self._name] = x
        return td


class Embedding(LayerBase):
    """
    A lookup table that stores embeddings of fixed dictionary and size.

    This layer stores embeddings for a fixed dictionary of indices, and retrieves
    them based on input indices. The embeddings are initialized using an orthogonal
    initialization and then scaled to have a maximum absolute value of 0.1.

    The shape of the output embeddings is [num_indices, embedding_dim], where
    num_indices can vary depending on the forward pass. Child layers should not
    be sensitive to changes in the first dimension.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(
        self,
        name: str,
        num_embeddings: int,
        embedding_dim: int,
        sources: list[str] | None = None,
        nn_params: NNParams | None = None,
    ):
        super().__init__(name, sources, nn_params)
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

    def _make_net(self):
        # output shape [0] is the number of indices used in the forward pass which can change
        # no child layer should be sensitive to this dimension
        self._out_tensor_shape = torch.Size([0, self._embedding_dim])

        net = nn.Embedding(self._num_embeddings, self._embedding_dim)

        weight_limit = 0.1
        nn.init.orthogonal_(net.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(net.weight))
            net.weight.mul_(weight_limit / max_abs_value)

        return net


class Conv2d(ParamLayer):
    """
    Applies a 2D convolution over an input signal composed of several input channels.

    This class automatically calculates output dimensions based on input shape,
    kernel size, stride, and padding.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._set_conv_dims()
        return nn.Conv2d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())  # type: ignore

    def _set_conv_dims(self):
        """Calculate flattened width and height. This allows us to change obs width and height."""
        assert len(self._in_tensor_shapes[0]) == 3, "Conv2d input tensor shape must be 3d (ignoring batch dimension)"
        assert self._in_tensor_shapes[0][1] is not None and self._in_tensor_shapes[0][2] is not None

        assert self._nn_params.kernel_size is not None
        assert self._nn_params.stride is not None
        assert self._nn_params.out_channels is not None

        self._input_height = self._in_tensor_shapes[0][1]
        self._input_width = self._in_tensor_shapes[0][2]

        padding = self._nn_params.padding or 0

        self._output_height = (
            (self._input_height + 2 * padding - self._nn_params.kernel_size) / self._nn_params.stride
        ) + 1
        self._output_width = (
            (self._input_width + 2 * padding - self._nn_params.kernel_size) / self._nn_params.stride
        ) + 1

        if not self._output_height.is_integer() or not self._output_width.is_integer():
            raise ValueError(f"CNN {self._name} output dimensions must be integers. Adjust padding or kernel size.")

        self._output_height = int(self._output_height)
        self._output_width = int(self._output_width)

        self._out_tensor_shape = torch.Size([self._nn_params.out_channels, self._output_height, self._output_width])


class MaxPool1d(LayerBase):
    """
    Applies a 1D max pooling over an input signal.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.MaxPool1d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class MaxPool2d(LayerBase):
    """
    Applies a 2D max pooling over an input signal.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.MaxPool2d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class AdaptiveAvgPool1d(LayerBase):
    """
    Applies a 1D adaptive average pooling over an input signal.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.AdaptiveAvgPool1d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class AdaptiveAvgPool2d(LayerBase):
    """
    Applies a 2D adaptive average pooling over an input signal.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.AdaptiveAvgPool2d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class AdaptiveMaxPool1d(LayerBase):
    """
    Applies a 1D adaptive max pooling over an input signal.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.AdaptiveMaxPool1d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class AdaptiveMaxPool2d(LayerBase):
    """
    Applies a 2D adaptive max pooling over an input signal.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.AdaptiveMaxPool2d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class AvgPool1d(LayerBase):
    """
    Applies a 1D average pooling over an input signal.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.AvgPool1d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class AvgPool2d(LayerBase):
    """
    Applies a 2D average pooling over an input signal.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.AvgPool2d(self._in_tensor_shapes[0][0], **self._nn_params.to_dict())


class Dropout(LayerBase):
    """
    Randomly zeroes some of the elements of the input tensor with probability p.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.Dropout(**self._nn_params.to_dict())


class Dropout2d(LayerBase):
    """
    Randomly zero out entire channels of the input tensor with probability p.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.Dropout2d(**self._nn_params.to_dict())


class AlphaDropout(LayerBase):
    """
    Applies Alpha Dropout, which maintains the mean and variance of the inputs.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.AlphaDropout(**self._nn_params.to_dict())


class BatchNorm1d(LayerBase):
    """
    Applies Batch Normalization over a 2D or 3D input.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.BatchNorm1d(self._in_tensor_shapes[0][0], **self._nn_params)


class BatchNorm2d(LayerBase):
    """
    Applies Batch Normalization over a 4D input.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.BatchNorm2d(self._in_tensor_shapes[0][0], **self._nn_params)


class Flatten(LayerBase):
    """
    Flattens a contiguous range of dimensions into a single dimension.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = torch.Size([prod(self._in_tensor_shapes[0])])
        return nn.Flatten()


class Identity(LayerBase):
    """
    A placeholder identity layer that returns the input tensor unchanged.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def _make_net(self):
        self._out_tensor_shape = self._in_tensor_shapes[0]
        return nn.Identity()
