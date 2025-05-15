import math
from typing import Any, Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing_extensions import override

from metta.agent.lib.metta_layer import LayerBase


class BilinearNetwork(nn.Module):
    """
    A custom bilinear network implementation with improved performance.

    This neural network module implements a bilinear transformation between hidden states
    and action embeddings, followed by optional MLP processing. It uses efficient tensor
    operations with einsum, which provides significantly better performance (approximately 10x)
    compared to using nn.Bilinear directly.

    The network computes bilinear interactions between hidden states and action embeddings,
    applies specified activation functions, and processes through an optional MLP to produce
    final outputs.
    """

    def __init__(self, W, bias, relu=None, tanh=None, mlp=None):
        """
        Initialize the BilinearNetwork with the necessary parameters and submodules.

        Args:
            W: Parameter tensor of shape [output_dim, hidden_dim, embed_dim] for the bilinear transformation
            bias: Parameter tensor of shape [output_dim] for bilinear output bias
            relu: Optional ReLU activation module (default: None)
            tanh: Optional Tanh activation module (default: None)
            mlp: Optional MLP module for further processing bilinear outputs (default: None)
        """
        super().__init__()

        # Initialize bilinear parameters
        self.W = W
        self.bias = bias

        # Activation functions
        self.relu = relu
        self.tanh = tanh

        # Final MLP layers
        self.mlp = mlp

    def forward(self, hidden_states, action_embeds):
        """
        Forward pass for the bilinear network.

        Computes a bilinear transformation between hidden states and action embeddings,
        followed by optional activation and MLP processing.

        Args:
            hidden_states: Tensor of shape [batch_size, hidden_dim] containing agent state
            action_embeds: Tensor of shape [batch_size, embed_dim] containing action embeddings

        Returns:
            Tensor containing processed bilinear outputs, either raw scores or processed through MLP
        """
        # Custom bilinear transformation using einsum for better performance
        # Equivalent to: x1 @ W @ x2.T + bias
        bilinear_output = torch.einsum("bi,oij,bj->bo", hidden_states, self.W, action_embeds) + self.bias

        # Apply activation if available
        if self.relu is not None:
            bilinear_output = self.relu(bilinear_output)

        # Apply MLP if available
        if self.mlp is not None:
            return self.mlp(bilinear_output)

        return bilinear_output


class MettaActorBig(LayerBase):
    """
    Implements a bilinear interaction layer followed by an MLP for action selection.

    This layer combines agent state and action embeddings through a bilinear interaction,
    followed by a multi-layer perceptron (MLP) to produce action logits. The implementation
    uses efficient tensor operations with einsum for the bilinear interaction, which is
    significantly faster than using nn.Bilinear directly.

    The layer works by:
    1) Taking hidden state and action embeddings as inputs
    2) Computing bilinear interactions between them
    3) Applying activation functions and an MLP
    4) Producing logits for each possible action

    This implementation replicates functionality that could be achieved by composing multiple
    simpler layers, but is optimized for performance.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, mlp_hidden_dim=512, bilinear_output_dim=32, **cfg):
        """
        Initialize the MettaActorBig layer.

        Args:
            mlp_hidden_dim: Dimension of the hidden layer in the MLP (default: 512)
            bilinear_output_dim: Output dimension of the bilinear transformation (default: 32)
            **cfg: Additional configuration parameters for the base layer
        """
        super().__init__(**cfg)
        self.mlp_hidden_dim = mlp_hidden_dim  # this is hardcoded for a two layer MLP
        self.bilinear_output_dim = bilinear_output_dim

    @override
    def _make_net(self) -> nn.Module:
        """
        Create and initialize the bilinear network module.

        This method constructs a custom BilinearNetwork that combines a manually implemented
        bilinear layer (for performance) with an MLP. The resulting network processes two
        inputs: hidden states and action embeddings.

        Returns:
            A PyTorch module implementing the bilinear network architecture
        """
        self.hidden = self._in_tensor_shapes[0][0]  # input_1 dim
        self.embed_dim = self._in_tensor_shapes[1][1]  # input_2 dim (_action_embeds_)

        # nn.Bilinear but hand written as nn.Parameters. As of 4-23-25, this is 10x faster than using nn.Bilinear.
        self.W = nn.Parameter(torch.Tensor(self.bilinear_output_dim, self.hidden, self.embed_dim))
        self.bias = nn.Parameter(torch.Tensor(self.bilinear_output_dim))
        self._init_weights()

        self._relu = nn.ReLU()
        self._tanh = nn.Tanh()

        self._MLP = nn.Sequential(
            nn.Linear(self.bilinear_output_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

        # Return the BilinearNetwork with initialized parameters
        return BilinearNetwork(W=self.W, bias=self.bias, relu=self._relu, tanh=self._tanh, mlp=self._MLP)

    def _init_weights(self):
        """
        Initialize weights using Kaiming (He) initialization.

        This method initializes the bilinear weights (W) and bias with uniform
        distribution scaled based on the input dimension.
        """
        bound = 1 / math.sqrt(self.hidden) if self.hidden > 0 else 0
        nn.init.uniform_(self.W, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @override
    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for the MettaActorBig layer.

        Processes hidden states and action embeddings through the bilinear network
        to produce action logits.

        Args:
            data: dict containing input tensors with keys matching source names

        Returns:
            Updated dict with action logits added under this layer's name
        """
        hidden = data[self._sources[0]["name"]]  # Shape: [B*TT, hidden]
        action_embeds = data[self._sources[1]["name"]]  # Shape: [B*TT, num_actions, embed_dim]

        B_TT = hidden.shape[0]
        num_actions = action_embeds.shape[1]

        # input_1: [B*TT, hidden] -> [B*TT * num_actions, hidden]
        # input_2: [B*TT, num_actions, embed_dim] -> [B*TT * num_actions, embed_dim]
        hidden_reshaped = repeat(hidden, "b h -> b a h", a=num_actions)  # shape: [B*TT, num_actions, hidden]
        hidden_reshaped = rearrange(hidden_reshaped, "b a h -> (b a) h")  # shape: [N, H]
        action_embeds_reshaped = rearrange(action_embeds, "b a e -> (b a) e")  # shape: [N, E]

        # Perform bilinear operation  h W e -> k for each B * num_actions = N
        query = torch.einsum("n h, k h e -> n k e", hidden_reshaped, self.W)  # Shape: [N, K, E]
        query = self._tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)  # Shape: [N, K]

        biased_scores = scores + self.bias.reshape(1, -1)  # Shape: [N, K]

        activated_scores = self._relu(biased_scores)  # Shape: [N, K]

        mlp_output = self._MLP(activated_scores)  # Shape: [N, 1]

        # Reshape MLP output back to sequence and action dimensions
        action_logits = mlp_output.reshape(B_TT, num_actions)  # Shape: [B*TT, num_actions]

        data[self._name] = action_logits
        return data


class MettaActorSingleHead(LayerBase):
    """
    Implements a simplified bilinear interaction layer for action selection.

    This class is a lighter version of MettaActorBig, using a single bilinear interaction
    without the additional MLP. It directly computes action logits from hidden state and
    action embeddings through an efficient bilinear operation implemented with einsum.

    The layer works by:
    1) Taking hidden state and action embeddings as inputs
    2) Computing a direct bilinear interaction between them
    3) Applying a tanh activation and adding bias
    4) Producing logits for each possible action

    This implementation is more efficient than MettaActorBig when a simpler action selection
    mechanism is sufficient, while maintaining the performance benefits of custom einsum operations
    over standard nn.Bilinear.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, **cfg):
        """
        Initialize the MettaActorSingleHead layer.

        Args:
            **cfg: Configuration parameters for the base layer
        """
        super().__init__(**cfg)

    @override
    def _make_net(self) -> nn.Module:
        """
        Create and initialize the simplified bilinear network module.

        This method constructs a BilinearNetwork with a single output dimension and
        no MLP, optimized for simpler action selection tasks.

        Returns:
            A PyTorch module implementing the simplified bilinear network
        """
        self.hidden = self._in_tensor_shapes[0][0]  # input_1 dim
        self.embed_dim = self._in_tensor_shapes[1][1]  # input_2 dim (_action_embeds_)

        # nn.Bilinear but hand written as nn.Parameters. As of 4-23-25, this is 10x faster than using nn.Bilinear.
        self.W = nn.Parameter(torch.Tensor(1, self.hidden, self.embed_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self._tanh = nn.Tanh()
        self._init_weights()

        # Return the simplified BilinearNetwork with initialized parameters
        return BilinearNetwork(W=self.W, bias=self.bias, tanh=self._tanh)

    def _init_weights(self):
        """
        Initialize weights using Kaiming (He) initialization.

        This method initializes the bilinear weights (W) and bias with uniform
        distribution scaled based on the input dimension.
        """
        bound = 1 / math.sqrt(self.hidden) if self.hidden > 0 else 0
        nn.init.uniform_(self.W, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @override
    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for the MettaActorSingleHead layer.

        Processes hidden states and action embeddings through the simplified
        bilinear network to produce action logits.

        Args:
            data: dict containing input tensors with keys matching source names

        Returns:
            Updated dict with action logits added under this layer's name
        """
        hidden = data[self._sources[0]["name"]]  # Shape: [B*TT, hidden]
        action_embeds = data[self._sources[1]["name"]]  # Shape: [B*TT, num_actions, embed_dim]

        B_TT = hidden.shape[0]
        num_actions = action_embeds.shape[1]

        # Reshape inputs similar to Rev2 for bilinear calculation
        # input_1: [B*TT, hidden] -> [B*TT * num_actions, hidden]
        # input_2: [B*TT, num_actions, embed_dim] -> [B*TT * num_actions, embed_dim]
        hidden_reshaped = repeat(hidden, "b h -> b a h", a=num_actions)  # shape: [B*TT, num_actions, hidden]
        hidden_reshaped = rearrange(hidden_reshaped, "b a h -> (b a) h")  # shape: [N, H]
        action_embeds_reshaped = rearrange(action_embeds, "b a e -> (b a) e")  # shape: [N, E]

        # Perform bilinear operation using einsum
        # Perform bilinear operation  h W e -> k for each B * num_actions = N
        query = torch.einsum("n h, k h e -> n k e", hidden_reshaped, self.W)  # Shape: [N, K, E]
        query = self._tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)  # Shape: [N, K]

        # Add bias
        biased_scores = scores + self.bias.reshape(1, -1)  # Shape: [N, K]

        action_logits = biased_scores.reshape(B_TT, num_actions)  # Shape: [B*TT, num_actions]

        data[self._name] = action_logits
        return data
