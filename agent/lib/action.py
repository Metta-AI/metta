import hashlib
import numpy as np
import torch

from tensordict import TensorDict

import agent.lib.nn_layer_library as nn_layer_library
import agent.lib.metta_layer as metta_layer

# '''Action classes expect two input sources: a feature representation vector and a action hash vector.'''

# class ActionType(nn_layer_library.Bilinear):
#     def __init__(self, **cfg):
#         super().__init__(**cfg)

#     def _forward(self, td: TensorDict):
#         # for source in self._input_source:
#         #     #rewrite this to go off of input sources in order
#         #     source_name = source['source_name']
#         #     if source_name == '_action_type_embeds_':
#         #         action_embeds = td[source_name]
#         #     else:
#         #         state_features = td[source_name]

#         input_1 = td[self._input_source[0]]
#         input_2 = td[self._input_source[1]]

#         # state_flat, action_flat, num_actions, N = shape_action_embed(input_1, input_2)
#         td[self._name] = self.bilinear(input_1, input_2)

#         # td[self._name] = logits_flat.view(N, num_actions) # Reshape to [N, num_actions]

#         return td

# class ActionParam(nn_layer_library.Bilinear):
#     def __init__(self, **cfg):
#         super().__init__(**cfg)

#     def _forward(self, td: TensorDict):
#         for source in self._input_source:
#             source_name = source['source_name']
#             if source_name == '_action_param_embeds_':
#                 action_embeds = td[source_name]
#             else:
#                 state_features = td[source_name]

#         state_flat, action_flat, num_actions, N = shape_action_embed(state_features, action_embeds)
#         logits_flat = self.bilinear(state_flat, action_flat)

#         td[self._name] = logits_flat.view(N, num_actions) # Reshape to [N, num_actions]

#         return td

class ActionEmbedding(nn_layer_library.Embedding):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self._reserved_action_embeds = {} 
        self._out_tensor_shape = [self._nn_params['embedding_dim']]
        self.num_actions = 0 
        # num_actions to be updated at runtime by the size of the active indices
        # we then make num_actions the batch dimension

    # see if there is a way to make unused embeddings orthogonal to the used ones

    def embed_strings(self, actions_list):
        # each time we run this, we update the metta_agent object's (the policy's) known action strings and associated indices

        # convert the actions_dict into a list of strings
        string_list = []
        for action_name, max_arg_count in actions_list:
            for i in range(max_arg_count):
                string_list.append(f"{action_name}_{i}")

        # for each action string, if it's not already in the reserved_action_embeds, add it and give it an index
        for action_type in string_list:
            if action_type not in self._reserved_action_embeds:
                embedding_index = len(self._reserved_action_embeds) + 1 # generate index for this string
                self._reserved_action_embeds[action_type] = embedding_index # update this component's known embeddings

        self.active_indices = torch.tensor([
            self._reserved_action_embeds[name]
            for name in string_list
        ])
        self.num_actions = len(self.active_indices)

    def _forward(self, td: TensorDict):
        td[self._name] = self._net(self.active_indices)
        return td
    
class ActionHash(metta_layer.LayerBase):
    def __init__(self, embedding_dim, **cfg):
        super().__init__(**cfg)
        self.action_embeddings = torch.tensor([])
        self._out_tensor_shape = [embedding_dim]
        self.num_actions = 0 # to be updated at runtime by the size of the embedding

    def embed_strings(self, actions_list):
        # convert the actions_dict into a list of strings
        string_list = []
        for action_name, max_arg_count in actions_list:
            for i in range(max_arg_count):
                string_list.append(f"{action_name}_{i}")

        self.action_embeddings = torch.tensor([
            self.embed_string(s)
            for s in string_list
        ], dtype=torch.float32)

        self.num_actions = self.action_embeddings.size(0)

    def embed_string(self, s):
        # Hash the string using SHA-256, which produces a 32-byte hash
        hash_object = hashlib.sha256(s.encode())
        hash_digest = hash_object.digest()

        # Convert hash bytes to a numpy array of floats
        # This example simply takes the first 'embedding_dim' bytes and scales them
        byte_array = np.frombuffer(hash_digest[:self._out_tensor_shape], dtype=np.uint8)
        embedding = byte_array / 255.0  # Normalize to range [0, 1]

        # Ensure the embedding is the right size
        if len(embedding) < self._out_tensor_shape:
            # If the hash is smaller than the needed embedding size, pad with zeros
            embedding = np.pad(embedding, (0, self._out_tensor_shape - len(embedding)), 'constant')

        return embedding

    def _forward(self, td: TensorDict):
        td[self._name] = self.action_embeddings
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