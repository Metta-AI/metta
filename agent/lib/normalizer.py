from __future__ import annotations

from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
from torch import nn
import torch

class FeatureListNormalizer(nn.Module):
    def __init__(self, feature_names, input_shape=(1,)):
        super().__init__()
        self._feature_names = feature_names
        self._norms_dict = nn.ModuleDict({
            **{
                k: RunningMeanStdInPlace(input_shape)
                for k in self._feature_names
            },
        })
        self._normalizers = [self._norms_dict[k] for k in self._feature_names]

    def forward(self, obs):
        with torch.no_grad():
            for fidx, norm in enumerate(self._normalizers):
                norm(obs[:, fidx, :])
