import torch
from torch import nn
import omegaconf
from tensordict import TensorDict

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
    def __init__(self, metta_agent, **cfg):
        super().__init__()
        cfg = omegaconf.OmegaConf.create(cfg)
        self.metta_agent = metta_agent
        self.metta_agent_components = metta_agent.components
        self.name = cfg.name
        self.input_source = cfg.input_source
        num_objects = len(self.metta_agent.grid_features)
        grid_features = self.metta_agent.grid_features

        obs_norm = torch.tensor([OBS_NORMALIZATIONS[k] for k in grid_features], dtype=torch.float32)
        obs_norm = obs_norm.view(1, num_objects, 1, 1)

        self.register_buffer('obs_norm', obs_norm)

    def set_input_size_and_initialize_layer(self):
        self.metta_agent_components[self.input_source].set_input_size_and_initialize_layer()
        self.input_size = self.metta_agent_components[self.input_source].output_size

        if self.output_size is None:
            self.output_size = self.input_size

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        self.metta_agent_components[self.input_source].forward(td)

        td[self.name] = td[self.input_source] / self.obs_norm

        return td
