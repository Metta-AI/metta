from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
from torch import nn
import torch
from tensordict import TensorDict

class FeatureListNormalizer(nn.Module):
    def __init__(self, metta_agent, **cfg):
        super().__init__()
        self.metta_agent = metta_agent
        self.cfg = cfg
        self.metta_agent_components = self.metta_agent.components
        self.name = self.cfg.name
        self.input_source = self.cfg.input_source
        self._feature_names = self.metta_agent.grid_features
        self.input_shape = self.metta_agent.obs_input_shape
        self._norms_dict = nn.ModuleDict({
            **{
                k: RunningMeanStdInPlace(self.input_shape)
                for k in self._feature_names
            },
        })
        self._normalizers = [self._norms_dict[k] for k in self._feature_names]

    def setup_layer(self):
        self.metta_agent_components[self.input_source].setup_layer()
        self.input_size = self.metta_agent_components[self.input_source].output_size

        if self.output_size is None:
            self.output_size = self.input_size

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        self.metta_agent_components[self.input_source].forward(td)

        with torch.no_grad():
            normalized_values = []
            for fidx, norm in enumerate(self._normalizers):
                normalized_values.append(norm(td[self.input_source][:, fidx, :, :]))
            td[self.name] = torch.stack(normalized_values, dim=1)

        return td