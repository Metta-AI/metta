import hashlib
import numpy as np
import torch

from tensordict import TensorDict

import agent.lib.nn_layer_library as nn_layer_library
import agent.lib.metta_layer as metta_layer

class ActionEmbedding(nn_layer_library.Embedding):
    def __init__(self, initialization='max_0_01', **cfg):
        super().__init__(**cfg)
        self._reserved_action_embeds = {} 
        self.num_actions = 0 
        # num_actions to be updated at runtime by the size of the active indices
        self._out_tensor_shape = [self.num_actions, self._nn_params['embedding_dim']]
        self.initialization = initialization
    def activate_actions(self, actions_list):
        # each time we run this, we update the metta_agent object's (the policy's) known action strings and associated indices

        # convert the actions_dict into a list of strings
        string_list = []
        for action_name, max_arg_count in actions_list:
            for i in range(max_arg_count + 1):
                string_list.append(f"{action_name}_{i}")

        # for each action string, if it's not already in the reserved_action_embeds, add it and give it an index
        for action_type in string_list:
            if action_type not in self._reserved_action_embeds:
                embedding_index = len(self._reserved_action_embeds) + 1 # generate index for this string
                self._reserved_action_embeds[action_type] = embedding_index # update this component's known embeddings

        device = next(self.parameters()).device
        self.active_indices = torch.tensor([
            self._reserved_action_embeds[name]
            for name in string_list
        ], device=device)
        self.num_actions = len(self.active_indices)

    def _forward(self, td: TensorDict):
        B = td['_batch_size_']
        TT = td['_TT_']
        td['_num_actions_'] = self.num_actions

        # below - get embeddings, unsqueeze the 0'th dimension, then expand to match the batch size
        td[self._name] = self._net(self.active_indices).unsqueeze(0).expand(B * TT, -1, -1)
        return td
    
class ActionHash(metta_layer.LayerBase):
    # This can't output hashes larger than 32
    def __init__(self, embedding_dim, min_value=-0.2, max_value=0.2, **cfg):
        super().__init__(**cfg)
        self.action_embeddings = torch.tensor([])
        self.num_actions = 0 # to be updated at runtime by the size of the embedding
        self.embedding_dim = embedding_dim
        self._out_tensor_shape = [self.num_actions, self.embedding_dim]
        self.value_min, self.value_max = min_value, max_value
        # Add a dummy parameter to track device
        self.register_buffer('dummy_param', torch.zeros(1))

    def activate_actions(self, actions_list):
        # convert the actions_dict into a list of strings
        string_list = []
        for action_name, max_arg_count in actions_list:
            for i in range(max_arg_count):
                string_list.append(f"{action_name}_{i}")
                
        # Use the dummy parameter to get the device
        device = self.dummy_param.device
        self.action_embeddings = torch.tensor([
            self.embed_string(s)
            for s in string_list
        ], dtype=torch.float32, device=device)

        self.num_actions = self.action_embeddings.size(0)

    def embed_string(self, s):
        hash_object = hashlib.sha256(s.encode())
        hash_digest = hash_object.digest()

        byte_array = np.frombuffer(hash_digest[:self.embedding_dim], dtype=np.uint8)
        # First normalize to [0,1]
        normalized = byte_array / 255.0
        # Then scale to desired range
        embedding = normalized * (self.value_max - self.value_min) + self.value_min

        # Ensure the embedding is the right size
        if len(embedding) < self.embedding_dim:
            # If the hash is smaller than the needed embedding size, pad with zeros
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), 'constant')

        return embedding

    def _forward(self, td: TensorDict):
        B = td['_batch_size_']
        TT = td['_TT_']
        td['_num_actions_'] = self.num_actions
        td[self._name] = self.action_embeddings.unsqueeze(0).expand(B * TT, -1, -1)
        return td



# def shape_action_embed(state_features, action_embeds):
#     """
#     Forward pass for scoring each action given state features.
#     Args:
#         state_features (torch.Tensor): Tensor of shape [N, feature_dim] where
#                                         N is the batch size.
#     Returns:
#         logits (torch.Tensor): Tensor of shape [N, num_actions], representing
#                                 the score for each action per state.
#     """
#     # state_features: [N, feature_dim] (we assume a vector)    
#     # action_embeds: [num_actions, embedding_dim]
#     num_actions = action_embeds.size(1) # should this be 0??
#     N = state_features.size(0)  # Batch size

#     # Expand state_features to prepare for pairing with each action.
#     # Add a dimension so state_features becomes [N, 1, feature_dim],
#     # then expand along that dimension to get [N, num_actions, feature_dim].
#     state_expanded = state_features.unsqueeze(1).expand(-1, num_actions, -1) 
#     # state_expanded: [N, num_actions, feature_dim]

#     # Similarly, expand the action embeddings so that they match the batch dimension.
#     # Start with [num_actions, embedding_dim], add a dimension to get [1, num_actions, embedding_dim],
#     # then expand to [N, num_actions, embedding_dim].
#     action_expanded = action_embeds.unsqueeze(0).expand(N, -1, -1)
#     # action_expanded: [N, num_actions, embedding_dim]

#     # Flatten both tensors so they can be passed to the bilinear layer.
#     # Merge the batch and action dimensions:
#     # state_flat: [N * num_actions, feature_dim]
#     state_flat = state_expanded.reshape(-1, state_features.size(1))
#     # action_flat: [N * num_actions, embedding_dim]
#     action_flat = action_expanded.reshape(-1, action_embeds.size(1))

#     # Apply the bilinear layer to each state-action pair.
#     # The bilinear layer processes pairs of [feature_dim] and [embedding_dim] to output a scalar.
#     # logits_flat: [N * num_actions, 1]
#     return state_flat, action_flat, num_actions, N



        
#     def get_stable_action_encoding(action_name, dim=8):
#         # Create a hash from action_name
#         hashed = int(hashlib.sha256(action_name.encode()).hexdigest(), 16) % (2**32)
#         gen = torch.Generator()
#         gen.manual_seed(hashed)
#         return torch.randn(dim, generator=gen)
    
#     # Get action names once
#     action_names = self.vecenv.driver_env.action_names()
#     # Create a tensor of shape [num_actions, embed_dim]
#     # ---- change this to be a one-time update function in action.py----
#     action_type_embeds = torch.stack([get_stable_action_encoding(action_name) for action_name in action_names])
#     action_param_embeds = torch.stack([get_stable_action_encoding(action_name) for action_name in action_names])

#     self.policy.components['_action_type_'].action_type_embeds = action_type_embeds
#     self.policy.components['_action_param_'].action_param_embeds = action_param_embeds