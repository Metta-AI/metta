import torch
from einops import repeat
from tensordict import TensorDict

import metta.agent.lib.nn_layer_library as nn_layer_library


class ActionEmbedding(nn_layer_library.Embedding):
    """
    Creates and manages embeddings for available actions in the environment.

    This class extends the base Embedding layer to specifically handle action embeddings
    in a reinforcement learning context. The MettaAgent's action registry manages the
    mapping between action names and indices, while this layer simply uses those indices.

    Key features:
    - Uses active indices provided by MettaAgent
    - Expands embeddings to match batch dimensions automatically
    - Stores the number of active actions in the TensorDict for other layers

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, initialization="max_0_01", **cfg):
        super().__init__(**cfg)
        self.num_actions = 0
        self.initialization = initialization
        self.register_buffer("active_indices", torch.tensor([], dtype=torch.long))

    def _forward(self, td: TensorDict):
        B_TT = td["_BxTT_"]
        td["_num_actions_"] = self.num_actions

        # get embeddings then expand to match the batch size
        td[self._name] = repeat(self._net(self.active_indices), "a e -> b a e", b=B_TT)
        return td
