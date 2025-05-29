import torch
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase

# ##ObservationNormalization
# These are approximate maximum values for each feature. Ideally they would be defined closer to their source,
# but here we are. If you add / remove a feature, you should add / remove the corresponding normalization.
OBS_NORMALIZATIONS = {
    "agent": 1,
    "agent:group": 10,
    "agent:hp": 30,
    "agent:frozen": 1,
    "agent:energy": 255,
    "agent:orientation": 1,
    "agent:shield": 1,
    "agent:color": 255,
    "inv:ore.red": 100,
    "inv:ore.blue": 100,
    "inv:ore.green": 100,
    "inv:battery.red": 100,
    "inv:battery.blue": 100,
    "inv:battery.green": 100,
    "inv:heart": 100,
    "inv:laser": 100,
    "inv:armor": 100,
    "inv:blueprint": 100,
    "wall": 1,
    "generator": 1,
    "mine": 1,
    "altar": 1,
    "armory": 1,
    "lasery": 1,
    "lab": 1,
    "factory": 1,
    "temple": 1,
    "last_action": 10,
    "temple:ready": 1,
    "last_action_argument": 10,
    "agent:kinship": 10,
    "hp": 30,
    "ready": 1,
    "converting": 1,
    "color": 10,
    "swappable": 1,
}


class ObservationNormalizer(LayerBase):
    """
    Normalizes observation features by dividing each feature by its approximate maximum value.

    This class scales observation features to a range of approximately [0, 1] by dividing
    each feature by predefined normalization values from the OBS_NORMALIZATIONS dictionary.
    Normalization helps stabilize neural network training by preventing features with large
    magnitudes from dominating the learning process.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, grid_features, **cfg):
        self._grid_features = grid_features
        super().__init__(**cfg)

    def _initialize(self):
        num_objects = len(self._grid_features)

        obs_norm = torch.tensor([OBS_NORMALIZATIONS[k] for k in self._grid_features], dtype=torch.float32)
        obs_norm = obs_norm.view(1, num_objects, 1, 1)

        self.register_buffer("obs_norm", obs_norm)

        self._out_tensor_shape = self._in_tensor_shapes[0].copy()

    def _forward(self, td: TensorDict):
        td[self._name] = td[self._sources[0]["name"]] / self.obs_norm
        return td
