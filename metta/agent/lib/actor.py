import torch
import torch.nn as nn
import math

from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase

class MettaActorBig(LayerBase):
    """
    Implements a bilinear interaction layer followed by an MLP with a lot of reshaping.
    It replicates what could be achieved by piecing together a number of other layers which would be more flexible.
    """
    def __init__(self, mlp_hidden_dim=512, bilinear_output_dim=32, **cfg):
        super().__init__(**cfg)
        self.mlp_hidden_dim = mlp_hidden_dim # this is hardcoded for a two layer MLP
        self.bilinear_output_dim = bilinear_output_dim

    def _make_net(self):
        self.hidden = self._in_tensor_shapes[0][0] # input_1 dim
        self.embed_dim = self._in_tensor_shapes[1][1] # input_2 dim (_action_embeds_)

        # nn.Bilinear but hand written as nn.Parameters. As of 4-23-25, this is 10x faster than using nn.Bilinear.
        self.W = nn.Parameter(torch.Tensor(self.bilinear_output_dim, self.hidden, self.embed_dim))
        self.bias = nn.Parameter(torch.Tensor(self.bilinear_output_dim))
        self._init_weights()

        self._relu = nn.ReLU()

        self._MLP = nn.Sequential(
            nn.Linear(self.bilinear_output_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def _init_weights(self):
        '''Kaiming (He) initialization'''
        bound = 1 / math.sqrt(self.hidden) if self.hidden > 0 else 0
        nn.init.uniform_(self.W, -bound, bound)
        if self.bias is not None:
             nn.init.uniform_(self.bias, -bound, bound)

    def _forward(self, td: TensorDict):
        input_1 = td[self._sources[0]["name"]] # Shape: [B*TT, hidden]
        input_2 = td[self._sources[1]["name"]] # Shape: [B*TT, num_actions, embed_dim]

        B_TT = input_1.shape[0]
        num_actions = input_2.shape[1]

        # input_1: [B*TT, hidden] -> [B*TT * num_actions, hidden]
        # input_2: [B*TT, num_actions, embed_dim] -> [B*TT * num_actions, embed_dim]
        input_1_reshaped = input_1.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, self.hidden)
        input_2_reshaped = input_2.reshape(-1, self.embed_dim)

        # Perform bilinear operation using einsum
        # einsum('n h, k h e, n e -> n k', ...) computes sum_{h,e} x1[n,h] * W[k,h,e] * x2[n,e] for each n, k
        # N = B_TT * num_actions, K = bilinear_output_dim
        scores = torch.einsum('n h, k h e, n e -> n k', input_1_reshaped, self.W, input_2_reshaped) # Shape: [N, K]

        # Add bias
        biased_scores = scores + self.bias.reshape(1, -1) # Shape: [N, K]

        # Apply activation
        activated_scores = self._relu(biased_scores) # Shape: [N, K]

        # Pass through MLP
        mlp_output = self._MLP(activated_scores) # Shape: [N, 1]

        # Reshape MLP output back to sequence and action dimensions
        action_logits = mlp_output.reshape(B_TT, num_actions) # Shape: [B*TT, num_actions]

        td[self._name] = action_logits
        return td
    
class MettaActorSingleHead(LayerBase):
    """
    Implements a bilinear interaction layer followed by an MLP with a lot of reshaping.
    It replicates what could be achieved by piecing together a number of other layers which would be more flexible.
    """
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self.hidden = self._in_tensor_shapes[0][0] # input_1 dim 
        self.embed_dim = self._in_tensor_shapes[1][1] # input_2 dim (_action_embeds_)

        # nn.Bilinear but hand written as nn.Parameters. As of 4-23-25, this is 10x faster than using nn.Bilinear.
        self.W = nn.Parameter(torch.Tensor(1, self.hidden, self.embed_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self._init_weights()

    def _init_weights(self):
        '''Kaiming (He) initialization'''
        bound = 1 / math.sqrt(self.hidden) if self.hidden > 0 else 0
        nn.init.uniform_(self.W, -bound, bound)
        if self.bias is not None:
             nn.init.uniform_(self.bias, -bound, bound)

    def _forward(self, td: TensorDict):
        input_1 = td[self._sources[0]["name"]] # Shape: [B*TT, hidden]
        input_2 = td[self._sources[1]["name"]] # Shape: [B*TT, num_actions, embed_dim]

        B_TT = input_1.shape[0]
        num_actions = input_2.shape[1]

        # Reshape inputs similar to Rev2 for bilinear calculation
        # input_1: [B*TT, hidden] -> [B*TT * num_actions, hidden]
        # input_2: [B*TT, num_actions, embed_dim] -> [B*TT * num_actions, embed_dim]
        input_1_reshaped = input_1.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, self.hidden)
        input_2_reshaped = input_2.reshape(-1, self.embed_dim)

        # Perform bilinear operation using einsum
        # einsum('n h, k h e, n e -> n k', ...) computes sum_{h,e} x1[n,h] * W[k,h,e] * x2[n,e] for each n, k
        # N = B_TT * num_actions, K = bilinear_output_dim
        scores = torch.einsum('n h, k h e, n e -> n k', input_1_reshaped, self.W, input_2_reshaped) # Shape: [N, K]

        # Add bias
        biased_scores = scores + self.bias.reshape(1, -1) # Shape: [N, K]

        action_logits = biased_scores.reshape(B_TT, num_actions) # Shape: [B*TT, num_actions]

        td[self._name] = action_logits
        return td
