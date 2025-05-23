import torch
import torch.nn as nn
from tensordict import TensorDict


class MettaModule(nn.Module):
    """Base class for all Metta computation modules.

    A MettaModule is a computation unit that reads from and writes to a TensorDict.
    The module's interface is defined by its in_keys (what it reads) and out_keys
    (what it writes). This explicit interface enables automatic dependency resolution,
    validation, and zero-setup testing.
    """

    def __init__(self, in_keys=None, out_keys=None, input_shapes=None, output_shapes=None):
        """Initialize a MettaModule with explicit input and output key specifications.

        Args:
            in_keys (list[str] | None): Keys in the TensorDict that this module will read from.
                If None, defaults to an empty list. Each key must exist in the TensorDict
                before this module can execute.
            out_keys (list[str] | None): Keys in the TensorDict that this module will write to.
                If None, defaults to an empty list. These keys must be unique across all
                modules in a ComponentContainer to prevent silent overwrites.
            input_shapes (dict[str, tuple] | None): Expected shapes for input tensors, excluding batch dimension.
                If None, no shape validation is performed. Example: {"input": (10,)}.
            output_shapes (dict[str, tuple] | None): Expected shapes for output tensors, excluding batch dimension.
                If None, no shape validation is performed. Example: {"output": (5,)}.
        """
        super().__init__()
        self.in_keys = in_keys or []
        self.out_keys = out_keys or []

        # Optional shape specifications for validation
        self.input_shapes = input_shapes or {}  # {key: shape}
        self.output_shapes = output_shapes or {}  # {key: shape}

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Execute the module's computation on the input TensorDict.

        Args:
            tensordict (TensorDict): Input data container. Must contain all in_keys.

        Returns:
            TensorDict: The same TensorDict with out_keys added/updated.

        Raises:
            ValueError: If input shapes don't match input_shapes specification.
            KeyError: If required in_keys are missing from tensordict.
        """
        # Validate input shapes if specified
        self.validate_shapes(tensordict)

        # Subclasses must implement actual computation
        return tensordict

    def validate_shapes(self, tensordict: TensorDict):
        """Validate input tensor shapes against specifications.
        This method is called automatically by forward() to ensure that the input tensors
        match the expected shapes. Note that this method always returns true if you don't
        specify input_shapes.

        Args:
            tensordict (TensorDict): Input data to validate.

        Raises:
            ValueError: If any tensor shape doesn't match its specification.
        """
        # Validate input shapes
        for key in self.in_keys:
            if key in self.input_shapes:
                expected_shape = self.input_shapes[key]
                actual_shape = tensordict[key].shape[1:]  # Skip batch dimension
                if actual_shape != expected_shape:
                    raise ValueError(f"Input shape mismatch for '{key}': expected {expected_shape}, got {actual_shape}")


class LinearModule(MettaModule):
    """A simple linear transformation module.

    This module demonstrates how to implement a concrete MettaModule with:
    - Explicit input/output keys
    - Shape validation
    - Actual computation
    """

    def __init__(self, in_features: int, out_features: int, in_key: str = "input", out_key: str = "output"):
        """Initialize a linear transformation module.

        Args:
            in_features (int): Size of input features
            out_features (int): Size of output features
            in_key (str): Key to read input from TensorDict
            out_key (str): Key to write output to TensorDict
        """
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
            input_shapes={in_key: (in_features,)},
            output_shapes={out_key: (out_features,)},
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Apply linear transformation to input tensor.

        Args:
            tensordict (TensorDict): Must contain in_key with shape (batch_size, in_features)

        Returns:
            TensorDict: Same tensordict with out_key added/updated with shape (batch_size, out_features)
        """
        # Validate input shapes
        self.validate_shapes(tensordict)

        # Apply linear transformation
        tensordict[self.out_keys[0]] = self.linear(tensordict[self.in_keys[0]])

        return tensordict


class ReLUModule(MettaModule):
    """A ReLU activation module that applies element-wise rectified linear unit function."""

    def __init__(self, in_key: str = "input", out_key: str = "output"):
        """Initialize a ReLU module.

        Args:
            in_key (str): Key to read input from TensorDict
            out_key (str): Key to write output to TensorDict
        """
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
        )
        self.relu = nn.ReLU()

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Apply ReLU activation to input tensor.

        Args:
            tensordict (TensorDict): Must contain in_key

        Returns:
            TensorDict: Same tensordict with out_key added/updated
        """
        tensordict[self.out_keys[0]] = self.relu(tensordict[self.in_keys[0]])
        return tensordict


class Conv2dModule(MettaModule):
    """A 2D convolution module that applies convolution over input tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        in_key: str = "input",
        out_key: str = "output",
    ):
        """Initialize a 2D convolution module.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            padding (int): Padding added to input
            in_key (str): Key to read input from TensorDict
            out_key (str): Key to write output to TensorDict
        """
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
        )
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Apply 2D convolution to input tensor.

        Args:
            tensordict (TensorDict): Must contain in_key with shape (batch_size, in_channels, height, width)

        Returns:
            TensorDict: Same tensordict with out_key added/updated
        """
        tensordict[self.out_keys[0]] = self.conv2d(tensordict[self.in_keys[0]])
        return tensordict


class FlattenModule(MettaModule):
    """A flatten module that flattens input tensors starting from a specified dimension."""

    def __init__(self, start_dim: int = 1, in_key: str = "input", out_key: str = "output"):
        """Initialize a flatten module.

        Args:
            start_dim (int): First dimension to flatten (default: 1, preserving batch dimension)
            in_key (str): Key to read input from TensorDict
            out_key (str): Key to write output to TensorDict
        """
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
        )
        self.flatten = nn.Flatten(start_dim=start_dim)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Apply flattening to input tensor.

        Args:
            tensordict (TensorDict): Must contain in_key

        Returns:
            TensorDict: Same tensordict with out_key added/updated
        """
        tensordict[self.out_keys[0]] = self.flatten(tensordict[self.in_keys[0]])
        return tensordict


class LayerNormModule(MettaModule):
    """A layer normalization module that normalizes inputs across features."""

    def __init__(self, normalized_shape: int, in_key: str = "input", out_key: str = "output"):
        """Initialize a layer normalization module.

        Args:
            normalized_shape (int): Size of features to normalize
            in_key (str): Key to read input from TensorDict
            out_key (str): Key to write output to TensorDict
        """
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
            input_shapes={in_key: (normalized_shape,)},
            output_shapes={out_key: (normalized_shape,)},
        )
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Apply layer normalization to input tensor.

        Args:
            tensordict (TensorDict): Must contain in_key with matching normalized_shape

        Returns:
            TensorDict: Same tensordict with out_key added/updated
        """
        self.validate_shapes(tensordict)
        tensordict[self.out_keys[0]] = self.norm(tensordict[self.in_keys[0]])
        return tensordict


class DropoutModule(MettaModule):
    """A dropout module that randomly zeroes some elements during training."""

    def __init__(self, p: float = 0.5, in_key: str = "input", out_key: str = "output"):
        """Initialize a dropout module.

        Args:
            p (float): Probability of an element to be zeroed
            in_key (str): Key to read input from TensorDict
            out_key (str): Key to write output to TensorDict
        """
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
        )
        self.dropout = nn.Dropout(p)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Apply dropout to input tensor.

        Args:
            tensordict (TensorDict): Must contain in_key

        Returns:
            TensorDict: Same tensordict with out_key added/updated
        """
        tensordict[self.out_keys[0]] = self.dropout(tensordict[self.in_keys[0]])
        return tensordict


class LSTMModule(MettaModule):
    """LSTM module that handles tensor reshaping and state management automatically.

    This module wraps a PyTorch LSTM with proper tensor shape handling, making it easier
    to integrate LSTMs into neural network policies. It handles reshaping inputs/outputs,
    manages hidden states, and ensures consistent tensor dimensions throughout the forward pass.

    The module processes tensors of shape [B*TT, ...], where:
    - B is the batch size
    - TT is the sequence length
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        in_key: str = "x",  # Match existing API
        out_key: str = "_lstm_",  # Match existing API
        hidden_key: str = "hidden",  # Match existing API
        state_key: str = "state",  # Match existing API
    ):
        """Initialize LSTM module.

        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            num_layers (int): Number of LSTM layers
            in_key (str): Key to read input from TensorDict (default: "x")
            out_key (str): Key to write output to TensorDict (default: "_lstm_")
            hidden_key (str): Key for hidden state in TensorDict (default: "hidden")
            state_key (str): Key for LSTM state in TensorDict (default: "state")
        """
        super().__init__(
            in_keys=[in_key, hidden_key],
            out_keys=[out_key, state_key],
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create LSTM network
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,  # We handle batch dimension manually
        )

        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Initialize bias to 1
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=1)  # Initialize weights orthogonally

    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        h_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        state = torch.cat([h_state, c_state], dim=0)  # [2*num_layers, batch_size, hidden_size]
        return state.permute(1, 0, 2)  # [batch_size, 2*num_layers, hidden_size]

    @torch.compile(disable=True)
    def forward(self, tensordict: TensorDict) -> TensorDict:
        x = tensordict[self.in_keys[0]]  # [B*TT, input_size]
        hidden = tensordict[self.in_keys[1]]  # [B*TT, input_size]
        state = tensordict.get(self.out_keys[1])  # [B, 2*num_layers, hidden_size] or None
        B = x.shape[0]
        assert hidden.shape == (B, self.input_size)

        # Handle state
        if state is not None:
            # Convert from [B, 2*num_layers, hidden_size] to [2*num_layers, B, hidden_size]
            state = state.permute(1, 0, 2)
            split_size = self.num_layers
            state = (state[:split_size], state[split_size:])
        else:
            # Create initial state with correct batch size
            state = self.get_initial_state(B, x.device)
            state = state.permute(1, 0, 2)
            state = (state[: self.num_layers], state[self.num_layers :])

        hidden = hidden.reshape(B, 1, self.input_size).transpose(0, 1)  # [1, B, input_size]
        hidden, state = self.lstm(hidden, state)
        hidden = hidden.transpose(0, 1).reshape(B, self.hidden_size)
        tensordict[self.out_keys[0]] = hidden

        # Save state as [B, 2*num_layers, hidden_size]
        if state is not None:
            state = tuple(s.detach() for s in state)
            state = torch.cat(state, dim=0)  # [2*num_layers, B, hidden_size]
            state = state.permute(1, 0, 2)  # [B, 2*num_layers, hidden_size]
            tensordict[self.out_keys[1]] = state
        return tensordict


if __name__ == "__main__":
    # Test successful case
    print("\n=== Testing successful case ===")
    module = LinearModule(in_features=10, out_features=5)
    td = TensorDict({"input": torch.randn(32, 10)}, batch_size=32)
    output = module(td)
    print(f"Input shape: {td['input'].shape}")
    print(f"Output shape: {td['output'].shape}")

    # Test shape mismatch
    print("\n=== Testing shape mismatch ===")
    try:
        bad_td = TensorDict({"input": torch.randn(32, 8)}, batch_size=32)  # Wrong input size
        module(bad_td)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test missing key
    print("\n=== Testing missing key ===")
    try:
        empty_td = TensorDict({}, batch_size=32)  # No input key
        module(empty_td)
    except KeyError as e:
        print(f"Caught expected error: {e}")

    # Test ReLU
    print("\n=== Testing ReLU ===")
    relu = ReLUModule()
    td = TensorDict({"input": torch.randn(32, 5)}, batch_size=32)
    output = relu(td)
    print(f"ReLU output shape: {td['output'].shape}")
