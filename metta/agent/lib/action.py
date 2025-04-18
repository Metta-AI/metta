import hashlib
import numpy as np
import torch

from tensordict import TensorDict

import metta.agent.lib.nn_layer_library as nn_layer_library
import metta.agent.lib.metta_layer as metta_layer

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
        # convert the actions_dict into a list of strings
        # string_list = []
        # for action_name, max_arg_count in actions_list:
        #     for i in range(max_arg_count):
        #         string_list.append(f"{action_name}_{i}")
                
        # Use the dummy parameter to get the device
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
