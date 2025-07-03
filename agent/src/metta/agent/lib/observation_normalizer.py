import torch
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


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

    def __init__(self, feature_normalizations, **cfg):
        self._feature_normalizations = feature_normalizations
        super().__init__(**cfg)

    def _initialize(self):
        obs_norm = torch.ones(max(self._feature_normalizations.keys()) + 1, dtype=torch.float32)
        for i, val in self._feature_normalizations.items():
            obs_norm[i] = val
        obs_norm = obs_norm.view(1, len(self._feature_normalizations), 1, 1)

        self.register_buffer("obs_norm", obs_norm)

        self._out_tensor_shape = self._in_tensor_shapes[0].copy()

    def _forward(self, td: TensorDict):
        td[self._name] = td[self._sources[0]["name"]] / self.obs_norm
        return td
