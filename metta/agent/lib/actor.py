import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase
from metta.agent.lib.metta_module import CustomShapeInferenceMixin, MettaDict, MettaModule, UniqueOutKeyMixin


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
        super().__init__(**cfg)
        self.mlp_hidden_dim = mlp_hidden_dim  # this is hardcoded for a two layer MLP
        self.bilinear_output_dim = bilinear_output_dim

    def _make_net(self):
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

    def _init_weights(self):
        """Kaiming (He) initialization"""
        bound = 1 / math.sqrt(self.hidden) if self.hidden > 0 else 0
        nn.init.uniform_(self.W, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _forward(self, td: TensorDict):
        hidden = td[self._sources[0]["name"]]  # Shape: [B*TT, hidden]
        action_embeds = td[self._sources[1]["name"]]  # Shape: [B*TT, num_actions, embed_dim]

        B_TT = hidden.shape[0]
        num_actions = action_embeds.shape[1]

        # input_1: [B*TT, hidden] -> [B*TT * num_actions, hidden]
        # input_2: [B*TT, num_actions, embed_dim] -> [B*TT * num_actions, embed_dim]
        hidden_reshaped = repeat(hidden, "b h -> b a h", a=num_actions)
        hidden_reshaped = rearrange(hidden_reshaped, "b a h -> (b a) h")
        action_embeds_reshaped = rearrange(action_embeds, "b a e -> (b a) e")

        # Perform bilinear operation  h W e -> k for each B * num_actions = N
        query = torch.einsum("n h, k h e -> n k e", hidden_reshaped, self.W)  # Shape: [N, K, E]
        query = self._tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)  # Shape: [N, K]

        biased_scores = scores + self.bias.reshape(1, -1)  # Shape: [N, K]

        activated_scores = self._relu(biased_scores)  # Shape: [N, K]

        mlp_output = self._MLP(activated_scores)  # Shape: [N, 1]

        # Reshape MLP output back to sequence and action dimensions
        action_logits = mlp_output.reshape(B_TT, num_actions)  # Shape: [B*TT, num_actions]

        td[self._name] = action_logits
        return td


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

    def _make_net(self):
        self.hidden = self._in_tensor_shapes[0][0]  # input_1 dim
        self.embed_dim = self._in_tensor_shapes[1][1]  # input_2 dim (_action_embeds_)

        # nn.Bilinear but hand written as nn.Parameters. As of 4-23-25, this is 10x faster than using nn.Bilinear.
        self.W = nn.Parameter(torch.Tensor(1, self.hidden, self.embed_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self._tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialization"""
        bound = 1 / math.sqrt(self.hidden) if self.hidden > 0 else 0
        nn.init.uniform_(self.W, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _forward(self, td: TensorDict):
        hidden = td[self._sources[0]["name"]]  # Shape: [B*TT, hidden]
        action_embeds = td[self._sources[1]["name"]]  # Shape: [B*TT, num_actions, embed_dim]

        B_TT = hidden.shape[0]
        num_actions = action_embeds.shape[1]

        # Reshape inputs similar to Rev2 for bilinear calculation
        # input_1: [B*TT, hidden] -> [B*TT * num_actions, hidden]
        # input_2: [B*TT, num_actions, embed_dim] -> [B*TT * num_actions, embed_dim]
        hidden_reshaped = repeat(hidden, "b h -> b a h", a=num_actions)
        hidden_reshaped = rearrange(hidden_reshaped, "b a h -> (b a) h")
        action_embeds_reshaped = rearrange(action_embeds, "b a e -> (b a) e")

        # Perform bilinear operation using einsum
        # Perform bilinear operation  h W e -> k for each B * num_actions = N
        query = torch.einsum("n h, k h e -> n k e", hidden_reshaped, self.W)  # Shape: [N, K, E]
        query = self._tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)  # Shape: [N, K]

        # Add bias
        biased_scores = scores + self.bias.reshape(1, -1)  # Shape: [N, K]

        action_logits = biased_scores.reshape(B_TT, num_actions)  # Shape: [B*TT, num_actions]

        td[self._name] = action_logits
        return td


class MettaActorBigModular(CustomShapeInferenceMixin, UniqueOutKeyMixin, MettaModule):
    """
    Modular rewrite of MettaActorBig using MettaModule/MettaDict pattern.
    Implements a bilinear interaction layer followed by an MLP for action selection.
    Accepts hidden state and action embeddings as input, outputs action logits.
    Provides shape inference for output shape given input shapes.
    """

    def infer_output_shape(self, input_shapes: list[list[int]]) -> list[int]:
        """
        Infers the output shape given the input shapes.
        Args:
            input_shapes: [[hidden_dim], [num_actions, embed_dim]]
        Returns:
            [num_actions]
        """
        if (
            input_shapes is None
            or not isinstance(input_shapes, list)
            or len(input_shapes) != 2
            or not all(isinstance(s, list) and len(s) > 0 for s in input_shapes)
        ):
            raise ValueError(
                "input_shapes must be a list of two non-empty lists: [[hidden_dim], [num_actions, embed_dim]]"
            )
        num_actions = input_shapes[1][0]
        return [num_actions]

    def __init__(
        self,
        in_keys: list[str],  # [hidden_key, action_embeds_key]
        out_keys: list[str],
        input_features_shape: list[list[int]],  # [[hidden_dim], [num_actions, embed_dim]]
        output_features_shape: list[int] = None,  # [num_actions]
        bilinear_output_dim: int = 32,
        mlp_hidden_dim: int = 512,
    ):
        if (
            input_features_shape is None
            or not isinstance(input_features_shape, list)
            or len(input_features_shape) != 2
            or not all(isinstance(s, list) and len(s) > 0 for s in input_features_shape)
        ):
            raise ValueError(
                "input_features_shape must be a list of two non-empty lists: [[hidden_dim], [num_actions, embed_dim]]"
            )
        if output_features_shape is None:
            output_features_shape = self.infer_output_shape(input_features_shape)
        super().__init__(in_keys, out_keys, input_features_shape=None, output_features_shape=output_features_shape)
        self.input_features_shape = input_features_shape
        self.bilinear_output_dim = bilinear_output_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        # Shapes
        if (
            self.input_features_shape is None
            or not isinstance(self.input_features_shape, list)
            or len(self.input_features_shape) != 2
            or not all(isinstance(s, list) and len(s) > 0 for s in self.input_features_shape)
        ):
            raise ValueError(
                "input_features_shape must be a list of two non-empty lists: [[hidden_dim], [num_actions, embed_dim]]"
            )
        self.hidden_dim = self.input_features_shape[0][0]
        self.embed_dim = self.input_features_shape[1][1]
        # Bilinear weights
        self.W = nn.Parameter(torch.empty(bilinear_output_dim, self.hidden_dim, self.embed_dim))
        self.bias = nn.Parameter(torch.empty(bilinear_output_dim))
        self._init_weights()
        self._relu = nn.ReLU()
        self._tanh = nn.Tanh()
        self._MLP = nn.Sequential(
            nn.Linear(bilinear_output_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def _init_weights(self):
        bound = 1 / math.sqrt(self.hidden_dim) if self.hidden_dim > 0 else 0
        nn.init.uniform_(self.W, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _compute(self, md: MettaDict) -> dict:
        hidden = md.td[self.in_keys[0]]  # [B*TT, hidden]
        action_embeds = md.td[self.in_keys[1]]  # [B*TT, num_actions, embed_dim]
        B_TT = hidden.shape[0]
        num_actions = action_embeds.shape[1]
        # Reshape
        hidden_reshaped = repeat(hidden, "b h -> b a h", a=num_actions)
        hidden_reshaped = rearrange(hidden_reshaped, "b a h -> (b a) h")
        action_embeds_reshaped = rearrange(action_embeds, "b a e -> (b a) e")
        # Bilinear interaction
        query = torch.einsum("n h, k h e -> n k e", hidden_reshaped, self.W)  # [N, K, E]
        query = self._tanh(query)
        scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)  # [N, K]
        biased_scores = scores + self.bias.reshape(1, -1)  # [N, K]
        activated_scores = self._relu(biased_scores)  # [N, K]
        mlp_output = self._MLP(activated_scores)  # [N, 1]
        logits = mlp_output.reshape(B_TT, num_actions)  # [B*TT, num_actions]
        return {self.out_key: logits}
