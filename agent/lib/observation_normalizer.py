import torch
import omegaconf
from tensordict import TensorDict

from agent.lib.metta_layer import LayerBase
# ##ObservationNormalization
# These are approximate maximum values for each feature. Ideally they would be defined closer to their source,
# but here we are. If you add / remove a feature, you should add / remove the corresponding normalization.
OBS_NORMALIZATIONS = {
    'agent': 1,
    'agent:group': 10,
    'agent:hp': 1,
    'agent:frozen': 1,
    'agent:energy': 255,
    'agent:orientation': 1,
    'agent:shield': 1,
    'agent:color': 255,
    'agent:inv:ore': 100,
    'agent:inv:battery': 100,
    'agent:inv:heart': 100,
    'agent:inv:laser': 100,
    'agent:inv:armor': 100,
    'agent:inv:blueprint': 100,
    'wall': 1,
    'wall:hp': 10,
    'generator': 1,
    'generator:hp': 30,
    'generator:ready': 1,
    'mine': 1,
    'mine:hp': 30,
    'mine:ready': 1,
    'altar': 1,
    'altar:hp': 30,
    'altar:ready': 1,
    'armory': 1,
    'armory:hp': 30,
    'armory:ready': 1,
    'lasery': 1,
    'lasery:hp': 30,
    'lasery:ready': 1,
    'lab': 1,
    'lab:hp': 30,
    'lab:ready': 1,
    'factory': 1,
    'factory:hp': 30,
    'factory:ready': 1,
    'temple': 1,
    'temple:hp': 30,
    'temple:ready': 1,
    'last_action': 10,
    'last_action_argument': 10,
    'agent:kinship': 10,
}

class ObservationNormalizer(LayerBase):
    def __init__(self, grid_features, **cfg):
        super().__init__(**cfg)

        num_objects = len(grid_features)

        obs_norm = torch.tensor([OBS_NORMALIZATIONS[k] for k in grid_features], dtype=torch.float32)
        obs_norm = obs_norm.view(1, num_objects, 1, 1)

        self.register_buffer('obs_norm', obs_norm)

    def _forward(self, td: TensorDict):
        td[self.name] = td[self.input_source] / self.obs_norm
        return td
