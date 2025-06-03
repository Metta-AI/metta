import torch
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase
from metta.agent.lib.metta_module import MettaDict, MettaModule, UniqueInKeyMixin, UniqueOutKeyMixin

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
    "converter": 1,
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
    "last_action": 10,
    "last_action_argument": 10,
    "agent:kinship": 10,
    "hp": 30,
    "converting": 1,
    "color": 10,
    "swappable": 1,
    "type_id": 10,
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


class MettaObsNormalizer(UniqueInKeyMixin, UniqueOutKeyMixin, MettaModule):
    def __init__(
        self,
        in_keys: list[str],
        out_keys: list[str],
        input_features_shape: list[int],
        output_features_shape: list[int],
        grid_features: list[str],
    ):
        super().__init__(in_keys, out_keys, input_features_shape, output_features_shape)
        self._grid_features = grid_features
        num_objects = len(self._grid_features)
        obs_norm = torch.tensor([OBS_NORMALIZATIONS[k] for k in self._grid_features], dtype=torch.float32)
        obs_norm = obs_norm.view(1, num_objects, 1, 1)
        self.register_buffer("obs_norm", obs_norm)

    def _compute(self, md: MettaDict) -> dict:
        return {self.out_key: md.td[self.in_key] / self.obs_norm}
