from typing import List

import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class EmbCriticConfig(ComponentConfig):
    """Variable depth MLP. You don't have to set input feats since it uses lazy linear.
    Don't set an output nonlinearity if it's used as an output head!"""

    in_key: str
    out_key: str
    out_features: int
    name: str = "emb_critic"
    in_features: int
    hidden_features: List[int]
    embeddings: int

    def make_component(self, env=None):
        return EmbCritic(config=self)


class EmbCritic(nn.Module):
    """A flexible MLP module using TensorDict."""

    def __init__(self, config: EmbCriticConfig):
        super().__init__()
        self.config = config

        self.linear1 = nn.Linear(self.config.in_features, self.config.hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.config.hidden_features, self.config.hidden_features)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(self.config.hidden_features, self.config.embeddings)
        self.out_key = self.config.out_key

        self.norm_factor = (self.config.embeddings) ** -0.5

    def forward(self, td: TensorDict):
        x = td[self.in_key]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = x * self.norm_factor
        x = x.sum(dim=-1)

        td[self.out_key] = x
        return td
