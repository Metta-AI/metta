import torch
from einops import repeat
from tensordict import TensorDict

import metta.agent.lib.nn_layer_library as nn_layer_library


class ActionEmbedding(nn_layer_library.Embedding):
    """
    Creates and manages embeddings for available actions in the environment.

    This class extends the base Embedding layer to specifically handle action embeddings
    in a reinforcement learning context. It maintains a dictionary mapping action names to
    embedding indices, and dynamically updates the set of active actions based on what's
    available in the current environment.

    Key features:
    - Maintains a mapping between action names (strings) and embedding indices
    - Dynamically activates subsets of actions when requested
    - Expands embeddings to match batch dimensions automatically
    - Stores the number of active actions in the TensorDict for other layers

    The activate_actions method should be called whenever the available actions in the
    environment change, providing the new set of action names and the target device.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

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
        """
        Updates the set of active action embeddings based on available actions.

        This method maintains a dictionary mapping action names to embedding indices.
        When new action names are encountered, they are assigned new indices.
        The method then creates a tensor of active indices on the specified device
        and updates the number of active actions.

        Args:
            action_names (list): List of action names (strings) available in the current environment
            device (torch.device): Device where the active_indices tensor should be stored
        """
        for action_name in action_names:
            if action_name not in self._reserved_action_embeds:
                embedding_index = len(self._reserved_action_embeds) + 1  # generate index for this string
                self._reserved_action_embeds[action_name] = embedding_index  # update this component's known embeddings

        self.active_indices = torch.tensor(
            [self._reserved_action_embeds[action_name] for action_name in action_names], device=device, dtype=torch.long
        )
        self.num_actions = len(self.active_indices)

    def _forward(self, td: TensorDict):
        B_TT = td["_BxTT_"]
        td["_num_actions_"] = self.num_actions

        # get embeddings then expand to match the batch size
        td[self._name] = repeat(self._net(self.active_indices), "a e -> b a e", b=B_TT)
        return td
