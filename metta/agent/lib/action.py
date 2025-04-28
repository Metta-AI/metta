import hashlib
import numpy as np
import torch
from torch import nn
import math

from tensordict import TensorDict

import metta.agent.lib.nn_layer_library as nn_layer_library
import metta.agent.lib.metta_layer as metta_layer
from metta.agent.lib.metta_layer import LayerBase


class ActionEmbedding(nn_layer_library.Embedding):
    def __init__(self, initialization='max_0_01', **cfg):
        super().__init__(**cfg)
        self._reserved_action_embeds = {} 
        self.num_actions = 0 
        # num_actions to be updated at runtime by the size of the active indices
        self._out_tensor_shape = [self.num_actions, self._nn_params['embedding_dim']]
        self.initialization = initialization
        self.register_buffer('active_indices', torch.tensor([], dtype=torch.long))
        
    def activate_actions(self, strings, device):
        ''' each time we run this, we update the metta_agent object's (the policy's) known action strings and associated indices'''
        for string in strings:
            if string not in self._reserved_action_embeds:
                embedding_index = len(self._reserved_action_embeds) + 1 # generate index for this string
                self._reserved_action_embeds[string] = embedding_index # update this component's known embeddings

        self.active_indices = torch.tensor([
            self._reserved_action_embeds[name]
            for name in strings
        ], device=device)
        self.num_actions = len(self.active_indices)

    def _forward(self, td: TensorDict):
        B_TT = td['_BxTT_']
        td['_num_actions_'] = self.num_actions

        # get embeddings, unsqueeze the 0'th dimension, then expand to match the batch size
        td[self._name] = self._net(self.active_indices).unsqueeze(0).expand(B_TT, -1, -1)
        
        return td
    
class ActionHash(metta_layer.LayerBase):
    ''' This can't output hashes with embedding_dim larger than 32 '''
    def __init__(self, embedding_dim, min_value=0, max_value=1, **cfg):
        super().__init__(**cfg)
        self.action_embeddings = torch.tensor([])
        self.num_actions = 0 # to be updated at runtime by the size of the embedding
        self.embedding_dim = embedding_dim
        self._out_tensor_shape = [self.num_actions, self.embedding_dim]
        self.min_value, self.max_value = min_value, max_value

    def activate_actions(self, strings, device):
        self.action_embeddings = torch.tensor([
            self.embed_string(s)
            for s in strings
        ], dtype=torch.float32, device=device)

        self.num_actions = self.action_embeddings.size(0)

    def embed_string(self, s):
        hash_object = hashlib.sha256(s.encode())
        hash_digest = hash_object.digest()

        byte_array = np.frombuffer(hash_digest[:self.embedding_dim], dtype=np.uint8)
        # First normalize to [0,1]
        normalized = byte_array / 255.0
        # Then scale to desired range
        embedding = normalized * (self.max_value - self.min_value) + self.min_value

        # Ensure the embedding is the right size
        if len(embedding) < self.embedding_dim:
            # If the hash is smaller than the needed embedding size, pad with zeros
            print(f"Padding embedding with 0s from {len(embedding)} to {self.embedding_dim}")
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), 'constant')

        return embedding

    def _forward(self, td: TensorDict):
        B_TT = td['_BxTT_']
        td['_num_actions_'] = self.num_actions
        td[self._name] = self.action_embeddings.unsqueeze(0).expand(B_TT, -1, -1)
        return td
    
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
        self.hidden = self._in_tensor_shape[0][0] # input_1 dim
        self.embed_dim = self._in_tensor_shape[1][1] # input_2 dim (_action_embeds_)

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
        input_1 = td[self._input_source[0]] # Shape: [B*TT, hidden]
        input_2 = td[self._input_source[1]] # Shape: [B*TT, num_actions, embed_dim]

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
        self.hidden = self._in_tensor_shape[0][0] # input_1 dim 
        self.embed_dim = self._in_tensor_shape[1][1] # input_2 dim (_action_embeds_)

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
        input_1 = td[self._input_source[0]] # Shape: [B*TT, hidden]
        input_2 = td[self._input_source[1]] # Shape: [B*TT, num_actions, embed_dim]

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

class MettaActorBig2(LayerBase):
    """
    Implements a bilinear interaction layer followed by an MLP with a lot of reshaping.
    It replicates what could be achieved by piecing together a number of other layers which would be more flexible.
    """
    def __init__(self, mlp_hidden_dim=512, bilinear_output_dim=32, **cfg):
        super().__init__(**cfg)
        self.mlp_hidden_dim = mlp_hidden_dim # this is hardcoded for a two layer MLP
        self.bilinear_output_dim = bilinear_output_dim

    def _make_net(self):
        self.hidden = self._in_tensor_shape[0][0] # input_1 dim
        self.embed_dim = self._in_tensor_shape[1][1] # input_2 dim (_action_embeds_)

        # nn.Bilinear but hand written as nn.Parameters. As of 4-23-25, this is 10x faster than using nn.Bilinear.
        self.W = nn.Parameter(torch.Tensor(self.bilinear_output_dim, self.hidden, self.embed_dim))
        self.bias = nn.Parameter(torch.Tensor(self.bilinear_output_dim))
        self._init_weights()

        self._relu = nn.ReLU()

        self._MLP = nn.Sequential(
            nn.Linear(self.bilinear_output_dim + self.hidden, self.mlp_hidden_dim),
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
        input_1 = td[self._input_source[0]] # Shape: [B*TT, hidden]
        input_2 = td[self._input_source[1]] # Shape: [B*TT, num_actions, embed_dim]

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

        # Concatenate the biased scores with the corresponding input representation

        biased_scores_cat_hidden = torch.cat([biased_scores, input_1], dim=-1)  # Shape: [N, K + hidden]

        # Apply activation
        activated_scores = self._relu(biased_scores_cat_hidden) # Shape: [N, K + hidden]

        # Pass through MLP
        mlp_output = self._MLP(activated_scores) # Shape: [N, 1]

        # Reshape MLP output back to sequence and action dimensions
        action_logits = mlp_output.reshape(B_TT, num_actions) # Shape: [B*TT, num_actions]

        td[self._name] = action_logits
        return td