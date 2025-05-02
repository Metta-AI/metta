import torch
from einops import repeat
from tensordict import TensorDict

import metta.agent.lib.nn_layer_library as nn_layer_library


class ActionEmbedding(nn_layer_library.Embedding):
    def __init__(self, initialization="max_0_01", **cfg):
        super().__init__(**cfg)
        self._reserved_action_embeds = {}
        self.num_actions = 0
        # delete this
        # # num_actions to be updated at runtime by the size of the active indices
        # self._out_tensor_shape = [self.num_actions, self._nn_params['embedding_dim']]
        self.initialization = initialization
        self.register_buffer("active_indices", torch.tensor([], dtype=torch.long))

    def activate_actions(self, action_names, device):
        """each time we run this, we update the metta_agent object's (the policy's) known action strings and associated
        indices"""
        # for string in full_action_names:
        for action_name in action_names:
            if action_name not in self._reserved_action_embeds:
                embedding_index = len(self._reserved_action_embeds) + 1  # generate index for this string
                self._reserved_action_embeds[action_name] = embedding_index  # update this component's known embeddings

        self.active_indices = torch.tensor(
            [self._reserved_action_embeds[action_name] for action_name in action_names], device=device
        )
        self.num_actions = len(self.active_indices)

    def _forward(self, td: TensorDict):
        B_TT = td["_BxTT_"]
        td["_num_actions_"] = self.num_actions

        # get embeddings then expand to match the batch size
        td[self._name] = repeat(self._net(self.active_indices), "a e -> b a e", b=B_TT)
        return td
