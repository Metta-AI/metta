import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class MettaActorBig(LayerBase):
    """
    Implements a bilinear interaction layer followed by an MLP with a lot of reshaping.
    It replicates what could be achieved by piecing together a number of other layers which would be more flexible.
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

        td[self._name] = action_logits
        return td


class MettaActorSingleHead(LayerBase):
    """
    Implements a linear followed by a bilinear interaction layer.
    It replicates what could be achieved by piecing together a number of other layers which would be more flexible.
    """

    def __init__(self, **cfg):
        super().__init__(**cfg)

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

        td[self._name] = action_logits
        return td


class PufferActor(LayerBase):
    """
    Implements a single Linear layer that is the same size as [hidden, num_actions]
    to replicate the Puffer small agent."""

    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self.hidden = self._in_tensor_shapes[0][0]  # input_1 dim
        self.embed_dim = self._in_tensor_shapes[1][1]  # input_2 dim (_action_embeds_)

    def _forward(self, td: TensorDict):
        hidden = td[self._sources[0]["name"]]  # Shape: [B*TT, hidden]
        action_embeds = td[self._sources[1]["name"]]  # Shape: [B*TT, num_actions, embed_dim]

        B_TT = hidden.shape[0]
        num_actions = action_embeds.shape[1]

        # Reshape inputs similar to Rev2 for bilinear calculation
        # input_1: [B*TT, hidden] -> [B*TT * num_actions, hidden]
        # input_2: [B*TT, num_actions, embed_dim] -> [B*TT * num_actions, embed_dim]
        hidden_reshaped = repeat(hidden, "b h -> b a h", a=num_actions)  # shape: [B*TT, num_actions, hidden]
        hidden_reshaped = rearrange(hidden_reshaped, "b a h -> (b a) h")  # shape: [N, H]
        action_embeds_reshaped = rearrange(action_embeds, "b a e -> (b a) e")  # shape: [N, E]

        action_logits = torch.einsum("nh,ne->n", hidden_reshaped, action_embeds_reshaped)  # Shape: [N]
        action_logits = rearrange(action_logits, "(b a) -> b a", b=B_TT, a=num_actions)  # Shape: [B*TT, num_actions]

        td[self._name] = action_logits
        return td
