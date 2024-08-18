
from __future__ import annotations
from typing import List

from torch import nn
import torch


from .lib.util import make_nn_stack, embed_strings

import numpy as np
import torch


from agent.lib.normalizer import FeatureListNormalizer
from .lib.util import make_nn_stack


class FeatureSetEncoder(nn.Module):
    def __init__(
            self,
            obs_space,
            obs_key: str,
            feature_names: List[str],
            normalize_features: bool,
            label_dim: int,
            output_dim: int,
            layers: int
        ):
        super().__init__()

        assert len(feature_names) == obs_space[obs_key].shape[0], \
            f"Number of feature names ({len(feature_names)}) must match" \
            f"the number of features in the observation space (" \
            f"{obs_space[obs_key].shape[0]})"

        self._obs_key = obs_key
        self._normalize_features = normalize_features
        self._feature_names = feature_names
        self._obs_shape = obs_space[obs_key].shape[1:] or (1,)
        self._normalizer = FeatureListNormalizer(feature_names, self._obs_shape)
        self._input_dim = np.prod(self._obs_shape)
        self._output_dim = output_dim
        self._label_dim = label_dim

        self._num_features = len(feature_names)

        self._labels_emb = embed_strings(self._feature_names, label_dim)
        self._labels_emb = self._labels_emb.unsqueeze(0)

        self.embedding_net = make_nn_stack(
            input_size=label_dim + self._input_dim,
            output_size=self._output_dim,
            hidden_sizes=[self._output_dim] * (layers - 1),
        )

    def forward(self, obs_dict):
        batch_size = obs_dict[self._obs_key].size(0)

        self._labels_emb = self._labels_emb.to(obs_dict[self._obs_key].device)
        obs = obs_dict[self._obs_key].view(batch_size, -1, *self._obs_shape).to(torch.float32)

        if self._normalize_features:
            self._normalizer(obs)

        labeled_obs = torch.cat([
            self._labels_emb.expand(batch_size, -1, -1),
            obs.view(batch_size, self._num_features, self._input_dim)
        ], dim=-1)

        x = torch.sum(self.embedding_net(labeled_obs), dim=1)
        return x

    def output_dim(self):
        return self._output_dim

class MultiFeatureSetEncoder(nn.Module):
    def __init__(self, obs_space, encoders_cfg, layers: int, output_dim: int):
        super().__init__()

        self.feature_set_encoders = nn.ModuleDict({
            name: FeatureSetEncoder(obs_space, name, **cfg)
            for name, cfg in encoders_cfg.items()
            if len(cfg.feature_names) > 0
        })

        self.merged_encoder = make_nn_stack(
            input_size=sum(encoder.output_dim() for encoder in self.feature_set_encoders.values()),
            output_size=output_dim,
            hidden_sizes=[output_dim] * (layers - 1),
        )

    def forward(self, obs_dict):
        x = torch.cat([encoder(obs_dict) for encoder in self.feature_set_encoders.values()], dim=1)
        x = self.merged_encoder(x)
        return x
