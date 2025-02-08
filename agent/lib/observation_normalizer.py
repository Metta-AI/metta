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
    def __init__(self, MettaAgent, **cfg):
        super().__init__()
        cfg = omegaconf.OmegaConf.create(cfg)
        self.MettaAgent = MettaAgent
        self.name = cfg.name
        self.input_source = cfg.input_source

        num_objects = len(self.MettaAgent.grid_features)

        obs_norm = torch.tensor([OBS_NORMALIZATIONS[k] for k in self.MettaAgent.grid_features], dtype=torch.float32)
        obs_norm = obs_norm.view(1, num_objects, 1, 1)

        self.register_buffer('obs_norm', obs_norm)

    def set_input_size_and_initialize_layer(self):
        if isinstance(self.input_source, omegaconf.listconfig.ListConfig):
            self.input_source = list(self.input_source)
            for src in self.input_source:
                self.MettaAgent.components[src].set_input_size_and_initialize_layer()

            self.input_size = sum(self.MettaAgent.components[src].output_size for src in self.input_source)
        else:
            self.MettaAgent.components[self.input_source].set_input_size_and_initialize_layer()
            self.input_size = self.MettaAgent.components[self.input_source].output_size

        if self.output_size is None:
            self.output_size = self.input_size

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        if isinstance(self.input_source, list):
            for src in self.input_source:
                self.MettaAgent.components[src].forward(td) 
        else:
            self.MettaAgent.components[self.input_source].forward(td)

        if isinstance(self.input_source, list):
            inputs = [td[src] for src in self.input_source]
            x = torch.cat(inputs, dim=-1)
            td[self.name] = x / self.obs_norm
        else:
            td[self.name] = td[self.input_source] / self.obs_norm

        return td
