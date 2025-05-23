import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
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
                # Allow for flexible batch dimension (None or any value)
                if expected_shape and expected_shape[0] is None:
                    if actual_shape[-len(expected_shape[1:]) :] != expected_shape[1:]:
                        raise ValueError(
                            f"Input shape mismatch for '{key}': expected {expected_shape}, got {actual_shape}"
                        )
                else:
                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"Input shape mismatch for '{key}': expected {expected_shape}, got {actual_shape}"
                        )


class LinearModule(MettaModule):
    """
    A MettaModule-compatible linear layer that applies a linear transformation to the input data.

    This module wraps PyTorch's nn.Linear with explicit input/output keys and shape validation.
    It processes tensors of shape [B*TT, ...] where B is the batch size and TT is the sequence length.

    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True
        in_key (str, optional): Key for the input tensor in the TensorDict. Default: "x"
        out_key (str, optional): Key for the output tensor in the TensorDict. Default: "out"
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        in_key: str = "x",
        out_key: str = "out",
    ):
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key],
            input_shapes={in_key: (in_features,)},
            output_shapes={out_key: (out_features,)},
        )
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass of the linear layer.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing the input tensor
                under the input_key. The input tensor should have shape [B*TT, in_features].

        Returns:
            TensorDict: Updated tensor dictionary with the output tensor under the output_key.
                The output tensor will have shape [B*TT, out_features].
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
            input_shapes={
                in_key: (input_size,),
                hidden_key: (hidden_size,),  # Hidden input should match hidden_size
            },
            output_shapes={
                out_key: (hidden_size,),
                state_key: (2 * num_layers, hidden_size),
            },
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
        """Forward pass of the LSTM module.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - in_key: Input tensor of shape [B, input_size]
                - hidden_key: Hidden state tensor of shape [B, hidden_size]

        Returns:
            TensorDict: Updated tensor dictionary with:
                - out_key: Output tensor of shape [B, hidden_size]
                - state_key: LSTM state tensor of shape [B, 2*num_layers, hidden_size]
        """
        self.validate_shapes(tensordict)

        x = tensordict[self.in_keys[0]]  # [B, input_size]
        hidden = tensordict[self.in_keys[1]]  # [B, hidden_size]
        B = x.shape[0]
        device = x.device

        # Prepare LSTM input: [seq_len=1, batch, input_size]
        x_seq = x.unsqueeze(0)  # [1, B, input_size]

        # Prepare initial hidden state: (h_0, c_0), each [num_layers, B, hidden_size]
        h_0 = (
            hidden.unsqueeze(0).expand(self.num_layers, B, self.hidden_size).contiguous()
        )  # [num_layers, B, hidden_size]
        c_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device, dtype=hidden.dtype)
        state = (h_0, c_0)

        # LSTM forward
        output, (h_n, c_n) = self.lstm(x_seq, state)
        output = output.squeeze(0)  # [B, hidden_size]
        tensordict[self.out_keys[0]] = output

        # Save state as [B, 2*num_layers, hidden_size]
        state_cat = torch.cat([h_n, c_n], dim=0)  # [2*num_layers, B, hidden_size]
        state_cat = state_cat.permute(1, 0, 2)  # [B, 2*num_layers, hidden_size]
        tensordict[self.out_keys[1]] = state_cat
        return tensordict


class ActorModule(MettaModule):
    """
    A MettaModule-compatible actor layer that implements a bilinear interaction followed by an MLP.

    This module combines agent state and action embeddings through a bilinear interaction,
    followed by a multi-layer perceptron (MLP) to produce action logits. The implementation
    uses efficient tensor operations with einsum for the bilinear interaction.

    Args:
        hidden_size (int): Size of the hidden state
        embed_dim (int): Size of the action embeddings
        mlp_hidden_dim (int, optional): Size of the hidden layer in the MLP. Default: 512
        bilinear_output_dim (int, optional): Size of the bilinear interaction output. Default: 32
        hidden_key (str, optional): Key for the hidden state tensor. Default: "hidden"
        action_embeds_key (str, optional): Key for the action embeddings tensor. Default: "action_embeds"
        output_key (str, optional): Key for the output logits tensor. Default: "action_logits"
    """

    def __init__(
        self,
        hidden_size: int,
        embed_dim: int,
        mlp_hidden_dim: int = 512,
        bilinear_output_dim: int = 32,
        hidden_key: str = "hidden",
        action_embeds_key: str = "action_embeds",
        output_key: str = "action_logits",
    ):
        super().__init__(
            in_keys=[hidden_key, action_embeds_key],
            out_keys=[output_key],
            input_shapes={
                hidden_key: (hidden_size,),
                action_embeds_key: (None, embed_dim),  # First dim is num_actions
            },
            output_shapes={output_key: (None,)},  # First dim is num_actions
        )

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.bilinear_output_dim = bilinear_output_dim

        # Bilinear interaction parameters
        self.W = nn.Parameter(torch.Tensor(bilinear_output_dim, hidden_size, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(bilinear_output_dim))

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(bilinear_output_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming (He) initialization."""
        bound = 1 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        nn.init.uniform_(self.W, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass of the actor layer.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - hidden_key: Hidden state tensor of shape [B*TT, hidden_size]
                - action_embeds_key: Action embeddings tensor of shape [B*TT, num_actions, embed_dim]

        Returns:
            TensorDict: Updated tensor dictionary with the output tensor under output_key.
                The output tensor will have shape [B*TT, num_actions].
        """
        # Validate input shapes
        self.validate_shapes(tensordict)

        # Get input tensors
        hidden = tensordict[self.in_keys[0]]  # Shape: [B*TT, hidden]
        action_embeds = tensordict[self.in_keys[1]]  # Shape: [B*TT, num_actions, embed_dim]

        B_TT = hidden.shape[0]
        num_actions = action_embeds.shape[1]

        # Reshape inputs for bilinear calculation
        hidden_reshaped = repeat(hidden, "b h -> b a h", a=num_actions)  # Shape: [B*TT, num_actions, hidden]
        hidden_reshaped = rearrange(hidden_reshaped, "b a h -> (b a) h")  # Shape: [N, H]
        action_embeds_reshaped = rearrange(action_embeds, "b a e -> (b a) e")  # Shape: [N, E]

        # Perform bilinear operation
        query = torch.einsum("n h, k h e -> n k e", hidden_reshaped, self.W)  # Shape: [N, K, E]
        query = self.tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)  # Shape: [N, K]

        # Add bias and apply ReLU
        biased_scores = scores + self.bias.reshape(1, -1)  # Shape: [N, K]
        activated_scores = self.relu(biased_scores)  # Shape: [N, K]

        # Apply MLP
        mlp_output = self.mlp(activated_scores)  # Shape: [N, 1]

        # Reshape output back to sequence and action dimensions
        action_logits = mlp_output.reshape(B_TT, num_actions)  # Shape: [B*TT, num_actions]

        # Store output in tensordict
        tensordict[self.out_keys[0]] = action_logits

        return tensordict


class MergeModule(MettaModule):
    """
    A MettaModule-compatible merge layer that combines multiple tensors from different sources.

    This module provides a framework for merging tensors from multiple sources in various ways.
    It handles tensor shape validation, optional slicing of source tensors, and tracking of
    input/output tensor dimensions.

    Args:
        merge_type (str): Type of merge operation to perform. One of:
            - "concat": Concatenate tensors along a specified dimension
            - "add": Add tensors element-wise
            - "subtract": Subtract second tensor from first element-wise
            - "mean": Compute element-wise mean of tensors
        merge_dim (int, optional): Dimension along which to merge tensors. Default: 1
        in_keys (list[str] | None, optional): Keys for input tensors in the TensorDict. Default: ["x1", "x2"]
        out_key (str, optional): Key for the output tensor in the TensorDict. Default: "out"
        slice_ranges (list[tuple[int, int]] | None, optional): Optional slice ranges for each input tensor.
            Each tuple (start, end) specifies the range to slice along merge_dim. Default: None
    """

    def __init__(
        self,
        merge_type: str,
        merge_dim: int = 1,
        in_keys: list[str] | None = None,
        out_key: str = "out",
        slice_ranges: list[tuple[int, int]] | None = None,
    ):
        if in_keys is None:
            in_keys = ["x1", "x2"]

        super().__init__(
            in_keys=in_keys,
            out_keys=[out_key],
        )

        self.merge_type = merge_type
        self.merge_dim = merge_dim
        self.slice_ranges = slice_ranges

        if merge_type not in ["concat", "add", "subtract", "mean"]:
            raise ValueError(f"Invalid merge_type: {merge_type}")

        if merge_type == "subtract" and len(in_keys) != 2:
            raise ValueError("Subtract merge requires exactly two input tensors")

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass of the merge layer.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing the input tensors
                under the in_keys. All input tensors should have compatible shapes.

        Returns:
            TensorDict: Updated tensor dictionary with the output tensor under out_key.
        """
        # Get input tensors
        inputs = [tensordict[key] for key in self.in_keys]

        # Apply optional slicing
        if self.slice_ranges is not None:
            inputs = [
                torch.narrow(x, dim=self.merge_dim, start=start, length=end - start)
                for x, (start, end) in zip(inputs, self.slice_ranges, strict=False)
            ]

        # Perform merge operation
        if self.merge_type == "concat":
            merged = torch.cat(inputs, dim=self.merge_dim)
        elif self.merge_type == "add":
            merged = sum(inputs)
        elif self.merge_type == "subtract":
            merged = inputs[0] - inputs[1]
        elif self.merge_type == "mean":
            merged = sum(inputs) / len(inputs)

        # Store output in tensordict
        tensordict[self.out_keys[0]] = merged

        return tensordict


class ObsModule(MettaModule):
    """
    A MettaModule-compatible observation shaper that handles observation tensor reshaping and validation.

    This module:
    1) Permutes input observations from [B, H, W, C] or [B, TT, H, W, C] to [..., C, H, W]
    2) Validates tensor shapes against expected environment observations
    3) Inserts batch size, TT, and B*TT into the tensor dict for other layers to use

    Args:
        obs_shape (tuple[int, int, int]): Expected observation shape (height, width, channels)
        num_objects (int): Number of objects in the observation
        in_key (str, optional): Key for the input observation tensor. Default: "x"
        out_key (str, optional): Key for the output tensor. Default: "obs"
        batch_size_key (str, optional): Key for storing batch size. Default: "_batch_size_"
        time_steps_key (str, optional): Key for storing time steps. Default: "_TT_"
        batch_time_key (str, optional): Key for storing batch*time. Default: "_BxTT_"
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        num_objects: int,
        in_key: str = "x",
        out_key: str = "obs",
        batch_size_key: str = "_batch_size_",
        time_steps_key: str = "_TT_",
        batch_time_key: str = "_BxTT_",
    ):
        super().__init__(
            in_keys=[in_key],
            out_keys=[out_key, batch_size_key, time_steps_key, batch_time_key],
        )

        self.obs_shape = list(obs_shape)  # Convert to list to avoid Omegaconf types
        self.num_objects = num_objects
        self.batch_size_key = batch_size_key
        self.time_steps_key = time_steps_key
        self.batch_time_key = batch_time_key

        # Output shape is [C, H, W] for conv layers
        self._out_tensor_shape = [self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]]

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass of the observation shaper.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing the observation tensor
                under in_key. The observation should have shape [B, H, W, C] or [B, TT, H, W, C].

        Returns:
            TensorDict: Updated tensor dictionary with:
                - out_key: Reshaped observation tensor of shape [B*TT, C, H, W]
                - batch_size_key: Batch size B as a tensor of shape [B]
                - time_steps_key: Time steps TT as a tensor of shape [B]
                - batch_time_key: B*TT as a tensor of shape [B]
        """
        x = tensordict[self.in_keys[0]]
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if tuple(x_shape[-space_n:]) != tuple(space_shape):
            expected_shape = f"[B(, T), {', '.join(str(dim) for dim in space_shape)}]"
            actual_shape = f"{list(x_shape)}"
            raise ValueError(
                f"Shape mismatch error:\n"
                f"x.shape: {x.shape}\n"
                f"self.obs_shape: {self.obs_shape}\n"
                f"Expected tensor with shape {expected_shape}\n"
                f"Got tensor with shape {actual_shape}\n"
                f"The last {space_n} dimensions should match {tuple(space_shape)}"
            )
        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError(
                f"Invalid input tensor dimensionality:\n"
                f"Expected tensor with {space_n + 1} or {space_n + 2} dimensions\n"
                f"Got tensor with {x_n} dimensions: {list(x_shape)}\n"
                f"Expected format: [batch_size(, time_steps), {', '.join(str(dim) for dim in space_shape)}]"
            )
        x = x.reshape(B * TT, *space_shape)
        x = x.float()
        x = self._permute(x)

        # Store batch size, time steps, and batchtime as tensors
        tensordict[self.batch_size_key] = torch.full((B,), B, dtype=torch.long, device=x.device)
        tensordict[self.time_steps_key] = torch.full((B,), TT, dtype=torch.long, device=x.device)
        tensordict[self.batch_time_key] = torch.full((B,), B * TT, dtype=torch.long, device=x.device)
        tensordict[self.out_keys[0]] = x
        return tensordict

    def _permute(self, x: torch.Tensor) -> torch.Tensor:
        """For compatibility with MPS, it throws an error on .permute()"""
        bs, h, w, c = x.shape
        x = x.contiguous().view(bs, h * w, c)
        x = x.transpose(1, 2)
        x = x.contiguous().view(bs, c, h, w)
        return x


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
