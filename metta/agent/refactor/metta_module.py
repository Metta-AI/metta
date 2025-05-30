import torch
import torch.nn as nn
from tensordict import TensorDict


class MettaModule(nn.Module):
    """Base class for TensorDict-aware modules using composition.

    Modules can work with both regular tensors and TensorDict inputs.
    Subclasses implement _forward_tensor() for tensor logic.
    """

    def __init__(self, in_keys=None, out_keys=None, input_shapes=None, output_shapes=None):
        """Initialize module with TensorDict key mappings.

        Args:
            in_keys: List of TensorDict keys for inputs
            out_keys: List of TensorDict keys for outputs
            input_shapes: Optional shape constraints for inputs
            output_shapes: Optional shape constraints for outputs
        """
        super().__init__()
        self.in_keys = in_keys or []
        self.out_keys = out_keys or []
        self.input_shapes = input_shapes or {}
        self.output_shapes = output_shapes or {}

    def forward(self, x):
        """Forward pass - handles both tensor and TensorDict inputs."""
        if isinstance(x, TensorDict):
            return self._forward_tensordict(x)
        else:
            return self._forward_tensor(x)

    def _forward_tensor(self, x):
        """Override this with your actual PyTorch logic.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        raise NotImplementedError

    def _forward_tensordict(self, td: TensorDict) -> TensorDict:
        """Process TensorDict input - extracts, processes, stores results.

        Args:
            td: Input TensorDict

        Returns:
            TensorDict with outputs stored at out_keys
        """
        if len(self.in_keys) == 1:
            inputs = td[self.in_keys[0]]
            outputs = self._forward_tensor(inputs)
            td[self.out_keys[0]] = outputs
        else:
            inputs = [td[key] for key in self.in_keys]
            outputs = self._forward_multi(*inputs)
            for i, key in enumerate(self.out_keys):
                td[key] = outputs[i] if isinstance(outputs, (tuple, list)) else outputs
        return td

    def _forward_multi(self, *inputs):
        """Override for modules with multiple tensor inputs.

        Args:
            *inputs: Variable number of input tensors

        Returns:
            Output tensor(s)
        """
        raise NotImplementedError("Multi-input modules must implement _forward_multi")


# Example Usage


class LinearModule(MettaModule):
    """Linear layer with TensorDict support.

    Example:
        # Tensor mode: works like nn.Linear
        linear = LinearModule(128, 64)
        output = linear(torch.randn(10, 128))

        # TensorDict mode: extracts from "input", stores in "output"
        td = TensorDict({"input": torch.randn(10, 128)}, batch_size=[10])
        td = linear(td)  # Result in td["output"]
    """

    def __init__(self, in_features: int, out_features: int, in_keys=None, out_keys=None):
        """Initialize linear module.

        Args:
            in_features: Input feature size
            out_features: Output feature size
            in_keys: TensorDict input keys (default: ["input"])
            out_keys: TensorDict output keys (default: ["output"])
        """
        super().__init__(in_keys=in_keys or ["input"], out_keys=out_keys or ["output"])
        self.linear = nn.Linear(in_features, out_features)  # Registered as submodule!

    def _forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation."""
        return self.linear(x)


class AttentionModule(MettaModule):
    """Multi-head attention with TensorDict support.

    Example:
        # TensorDict mode with multiple inputs
        attention = AttentionModule(512, 8)
        td = TensorDict({
            "query": torch.randn(10, 512),
            "key": torch.randn(10, 512),
            "value": torch.randn(10, 512)
        }, batch_size=[10])
        td = attention(td)  # Result in td["attention_output"]
    """

    def __init__(self, embed_dim: int, num_heads: int, in_keys=None, out_keys=None):
        """Initialize attention module.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            in_keys: TensorDict input keys (default: ["query", "key", "value"])
            out_keys: TensorDict output keys (default: ["attention_output"])
        """
        super().__init__(in_keys=in_keys or ["query", "key", "value"], out_keys=out_keys or ["attention_output"])
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def _forward_multi(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention to query, key, value tensors."""
        output, _ = self.attention(query, key, value)
        return output


# class ModularNetwork(MettaModule):
#     """Composable network of MettaModules with flexible data flow.

#     Modules are executed in sequence, with TensorDict flowing between them.
#     Each module can read from and write to different keys in the shared TensorDict.

#     Example:
#         # Create a simple pipeline
#         network = ModularNetwork([
#             LinearModule(128, 64, in_keys=["input"], out_keys=["hidden"]),
#             LinearModule(64, 32, in_keys=["hidden"], out_keys=["output"])
#         ])

#         # Use with TensorDict
#         td = TensorDict({"input": torch.randn(10, 128)}, batch_size=[10])
#         td = network(td)  # Result in td["output"]

#         # Or use with regular tensors (if network has single input/output)
#         output = network(torch.randn(10, 128))
#     """

#     def __init__(self, modules=None, in_keys=None, out_keys=None):
#         """Initialize modular network.

#         Args:
#             modules: List of MettaModules to compose
#             in_keys: Input keys for tensor mode (default: first module's in_keys)
#             out_keys: Output keys for tensor mode (default: last module's out_keys)
#         """
#         # Auto-detect in/out keys from first/last modules if not provided
#         if modules and in_keys is None:
#             in_keys = getattr(modules[0], "in_keys", [])
#         if modules and out_keys is None:
#             out_keys = getattr(modules[-1], "out_keys", [])

#         super().__init__(in_keys=in_keys, out_keys=out_keys)

#         # Register modules as submodules for proper PyTorch integration
#         self.modules_list = nn.ModuleList(modules or [])

#     def add_module_with_name(self, name: str, module: MettaModule):
#         """Add a named module to the network."""
#         self.modules_list.append(module)
#         # Also register with a name for easier access
#         setattr(self, name, module)

#     def _forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for tensor input - converts to TensorDict internally."""
#         if not self.in_keys or not self.out_keys:
#             raise ValueError("ModularNetwork needs in_keys and out_keys for tensor mode")

#         # Create temporary TensorDict
#         td = TensorDict({self.in_keys[0]: x}, batch_size=x.shape[:1])
#         td = self._forward_tensordict(td)
#         return td[self.out_keys[0]]

#     def _forward_tensordict(self, td: TensorDict) -> TensorDict:
#         """Process TensorDict through all modules in sequence."""
#         for module in self.modules_list:
#             td = module(td)  # Each module processes and updates the TensorDict
#         return td

#     def insert_module(self, index: int, module: MettaModule):
#         """Insert module at specific position."""
#         self.modules_list.insert(index, module)

#     def remove_module(self, index: int):
#         """Remove module at specific position."""
#         del self.modules_list[index]

#     def __len__(self):
#         """Number of modules in the network."""
#         return len(self.modules_list)

#     def __getitem__(self, index):
#         """Access module by index."""
#         return self.modules_list[index]
