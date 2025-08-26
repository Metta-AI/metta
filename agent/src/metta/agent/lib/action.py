import torch
from einops import repeat
from tensordict import TensorDict

import metta.agent.lib.nn_layer_library as nn_layer_library


class ActionEmbedding(nn_layer_library.Embedding):
    """The initialize_to_environment method should be called whenever the available actions in the
    environment change, providing the new set of action names and the target device.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy."""

    def __init__(self, initialization="max_0_01", **cfg):
        super().__init__(**cfg)
        self._reserved_action_embeds = {}
        self.num_actions = 0
        # delete this
        # # num_actions to be updated at runtime by the size of the active indices
        # self._out_tensor_shape = [self.num_actions, self._nn_params['embedding_dim']]
        self.initialization = initialization
        self.register_buffer("active_indices", torch.tensor([], dtype=torch.long))

    def initialize_to_environment(self, action_names, device):
        """Updates active action embeddings based on available actions.
        Assigns new indices for unseen actions and updates the active_indices tensor."""

        for action_name in action_names:
            if action_name not in self._reserved_action_embeds:
                embedding_index = len(self._reserved_action_embeds) + 1  # generate index for this string
                self._reserved_action_embeds[action_name] = embedding_index  # update this component's known embeddings

        self.active_indices = torch.tensor(
            [self._reserved_action_embeds[action_name] for action_name in action_names], device=device, dtype=torch.long
        )
        self.num_actions = len(self.active_indices)

    def _forward(self, td: TensorDict):
        B_TT = td.batch_size.numel()

        # get embeddings then expand to match the batch size
        td[self._name] = repeat(self._net(self.active_indices), "a e -> b a e", b=B_TT)
        return td
