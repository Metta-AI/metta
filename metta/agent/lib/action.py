import hashlib
import numpy as np
import torch
from torch import nn

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
        B = td['_batch_size_']
        TT = td['_TT_']
        td['_num_actions_'] = self.num_actions

        # get embeddings, unsqueeze the 0'th dimension, then expand to match the batch size
        td[self._name] = self._net(self.active_indices).unsqueeze(0).expand(B * TT, -1, -1).contiguous()
        
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
        # Add a dummy parameter to track device
        self.register_buffer('dummy_param', torch.zeros(1))

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
        B = td['_batch_size_']
        TT = td['_TT_']
        td['_num_actions_'] = self.num_actions
        td[self._name] = self.action_embeddings.unsqueeze(0).expand(B * TT, -1, -1)
        return td

class Als_Bilinear_Rev1(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        self.hidden = self._in_tensor_shape[0][0]
        self.embed_dim = self._in_tensor_shape[1][1]

        # Initialize learnable parameters
        self._W1 = nn.Linear(self.hidden, self.embed_dim)
        self._W2 = nn.Linear(self.hidden, self.embed_dim)
        self._W3 = nn.Linear(self.hidden, self.embed_dim)
        self._W4 = nn.Linear(self.hidden, self.embed_dim)
        self._W5 = nn.Linear(self.hidden, self.embed_dim)
        self._W6 = nn.Linear(self.hidden, self.embed_dim)
        self._W7 = nn.Linear(self.hidden, self.embed_dim)
        self._W8 = nn.Linear(self.hidden, self.embed_dim)
        self._W9 = nn.Linear(self.hidden, self.embed_dim)
        self._W10 = nn.Linear(self.hidden, self.embed_dim)
        self._W11 = nn.Linear(self.hidden, self.embed_dim)
        self._W12 = nn.Linear(self.hidden, self.embed_dim)
        self._W13 = nn.Linear(self.hidden, self.embed_dim)
        self._W14 = nn.Linear(self.hidden, self.embed_dim)
        self._W15 = nn.Linear(self.hidden, self.embed_dim)
        self._W16 = nn.Linear(self.hidden, self.embed_dim)
        self._W17 = nn.Linear(self.hidden, self.embed_dim)
        self._W18 = nn.Linear(self.hidden, self.embed_dim)
        self._W19 = nn.Linear(self.hidden, self.embed_dim)
        self._W20 = nn.Linear(self.hidden, self.embed_dim)
        self._W21 = nn.Linear(self.hidden, self.embed_dim)
        self._W22 = nn.Linear(self.hidden, self.embed_dim)
        self._W23 = nn.Linear(self.hidden, self.embed_dim)
        self._W24 = nn.Linear(self.hidden, self.embed_dim)
        self._W25 = nn.Linear(self.hidden, self.embed_dim)
        self._W26 = nn.Linear(self.hidden, self.embed_dim)
        self._W27 = nn.Linear(self.hidden, self.embed_dim)
        self._W28 = nn.Linear(self.hidden, self.embed_dim)
        self._W29 = nn.Linear(self.hidden, self.embed_dim)
        self._W30 = nn.Linear(self.hidden, self.embed_dim)
        self._W31 = nn.Linear(self.hidden, self.embed_dim)
        self._W32 = nn.Linear(self.hidden, self.embed_dim)
        
        
        # consider initializing with something finite if we get nets that can't get off the ground
        self._bias = nn.Parameter(torch.zeros(32))

        self._relu = nn.ReLU()

        self._MLP = nn.Sequential(
            nn.Linear(32, 512), # need to eventually get these from the config
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        return nn.Identity()  # We'll handle the computation in _forward
    
    def _forward(self, td: TensorDict):
        input_1 = td[self._input_source[0]] # _core_
        input_2 = td[self._input_source[1]] # _action_embeds_

        # Compute Q vectors
        q_1 = self._W1(input_1)
        q_2 = self._W2(input_1)
        q_3 = self._W3(input_1)
        q_4 = self._W4(input_1)
        q_5 = self._W5(input_1)
        q_6 = self._W6(input_1)
        q_7 = self._W7(input_1)
        q_8 = self._W8(input_1)
        q_9 = self._W9(input_1)
        q_10 = self._W10(input_1)
        q_11 = self._W11(input_1)
        q_12 = self._W12(input_1)
        q_13 = self._W13(input_1)
        q_14 = self._W14(input_1)
        q_15 = self._W15(input_1)
        q_16 = self._W16(input_1)
        q_17 = self._W17(input_1)
        q_18 = self._W18(input_1)
        q_19 = self._W19(input_1)
        q_20 = self._W20(input_1)
        q_21 = self._W21(input_1)
        q_22 = self._W22(input_1)
        q_23 = self._W23(input_1)
        q_24 = self._W24(input_1)
        q_25 = self._W25(input_1)
        q_26 = self._W26(input_1)
        q_27 = self._W27(input_1)
        q_28 = self._W28(input_1)
        q_29 = self._W29(input_1)
        q_30 = self._W30(input_1)
        q_31 = self._W31(input_1)
        q_32 = self._W32(input_1)
        

        # Stack Q vectors into a matrix
        Q = torch.stack([q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9, q_10, q_11, q_12, q_13, q_14, q_15, q_16, q_17, q_18, q_19, q_20, q_21, q_22, q_23, q_24, q_25, q_26, q_27, q_28, q_29, q_30, q_31, q_32], dim=1) # Shape: [B*TT, 32, embed_dim]

        # input_2 shape: [B*TT, num_actions, embed_dim]
        num_actions = input_2.shape[1] # Get num_actions dynamically
        B_TT = Q.shape[0] # This is B * TT

        # Perform batch matrix multiplication between Q and transposed input_2
        # Q: [B_TT, 8, embed_dim]
        # input_2.transpose(1, 2): [B_TT, embed_dim, num_actions]
        # scores_bmm: [B_TT, 8, num_actions]
        scores_bmm = torch.bmm(Q, input_2.transpose(1, 2))

        # Permute and reshape to [B_TT * num_actions, 8]
        # Permute: [B_TT, num_actions, 8]
        # Reshape: [B_TT * num_actions, 8]
        scores_reshaped = scores_bmm.permute(0, 2, 1).reshape(-1, 32)

        # Add bias
        biased_scores = self._relu(scores_reshaped + self._bias) # Shape: [B_TT * num_actions, 32]

        # should biased_scores go through a ReLU?

        # pass scores through MLP
        mlp_output = self._MLP(biased_scores) # Shape: [B_TT * num_actions, 1]

        # B = td["_batch_size_"]
        # TT = td["_TT_"] # Not needed if we reshape directly to B

        # Reshape scores from [B*TT*num_actions, 1] to [B, -1] (likely [B, TT*num_actions])
        # This matches the original code's final reshape operation.
        action_logits = mlp_output.reshape(B_TT, -1)

        td[self._name] = action_logits
        return td