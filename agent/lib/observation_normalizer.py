import torch
from torch import nn
from metta.agent.metta_agent import MettaAgent


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
    'agent:inv:r1': 5,
    'agent:inv:r2': 5,
    'agent:inv:r3': 5,
    'wall': 1,
    'wall:hp': 10,
    'generator': 1,
    'generator:hp': 30,
    'generator:r1': 30,
    'generator:ready': 1,
    'converter': 1,
    'converter:hp': 30,
    'converter:ready': 1,
    'altar': 1,
    'altar:hp': 30,
    'altar:ready': 1,
    'last_action': 10,
    'last_action_argument': 10,
    'agent:kinship': 10,
}

class ObservationNormalizer(nn.Module):
    def __init__(self, MettaAgent, grid_features: list[str]):
        super().__init__()
        self.MettaAgent = MettaAgent
        self.name = 'obs_normalizer'
        self.input_source = 'obs'
        self.output_size = None

        num_objects = len(grid_features)

        obs_norm = torch.tensor([OBS_NORMALIZATIONS[k] for k in grid_features], dtype=torch.float32)
        obs_norm = obs_norm.view(1, num_objects, 1, 1)

        self.register_buffer('obs_norm', obs_norm)

    def forward(self, obs):
        return obs / self.obs_norm
    
    def set_input_source_size(self):
        self.input_size = self.MettaAgent.components[self.input_source].get_out_size()
        if self.output_size is None:
            self.output_size = self.input_size

    def get_out_size(self):
        return self.output_size

    def initialize_layer(self):
        pass


