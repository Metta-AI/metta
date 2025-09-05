import torch
import torch.nn as nn
from tensordict import TensorDict


class ObservationNormalizer(nn.Module):
    """
    Normalizes observation features by dividing each feature by its approximate maximum value.

    This class scales observation features to a range of approximately [0, 1] by dividing
    each feature by predefined normalization values from the OBS_NORMALIZATIONS dictionary.
    Normalization helps stabilize neural network training by preventing features with large
    magnitudes from dominating the learning process.
    """

    def __init__(self, feature_normalizations, in_key="obs", out_key="obs_normalizer"):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self.update_normalization_factors(feature_normalizations)

    def forward(self, td: TensorDict):
        td[self.out_key] = td[self.in_key] / self.obs_norm
        return td

    def update_normalization_factors(self, features: dict[str, dict]):
        self.feature_normalizations = features
        obs_norm = torch.ones(max(self.feature_normalizations.keys()) + 1, dtype=torch.float32)
        for i, val in self.feature_normalizations.items():
            obs_norm[i] = val
        obs_norm = obs_norm.view(1, len(self.feature_normalizations), 1, 1)

        self.register_buffer("obs_norm", obs_norm)
