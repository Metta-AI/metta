import omegaconf
import torch
from tensordict import TensorDict
from torch import nn

from metta.agent.lib.metta_layer import LayerBase
from metta.agent.util.running_mean_std import RunningMeanStdInPlace


# this is not currently working
class FeatureListNormalizer(LayerBase):
    """
    Normalizes a list of features using running mean and standard deviation statistics.

    This layer applies feature-wise normalization to input observations, tracking the running
    mean and standard deviation of each feature independently. It creates a separate normalizer
    for each feature in the grid_features list provided by the metta_agent. Normalization is
    performed in-place using the RunningMeanStdInPlace class.

    The normalized features are stacked together and stored in the TensorDict under the layer's name.

    Note: This class is currently marked as not working.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, metta_agent, **cfg):
        super().__init__()
        cfg = omegaconf.OmegaConf.create(cfg)
        object.__setattr__(self, "metta_agent", metta_agent)
        self.cfg = cfg
        self.metta_agent_components = self.metta_agent.components
        self.name = self.cfg.name
        self.input_source = self.cfg.input_source
        self.output_size = None
        self._feature_names = self.metta_agent.grid_features
        self.input_shape = self.metta_agent.obs_input_shape
        self._norms_dict = nn.ModuleDict(
            {
                **{k: RunningMeanStdInPlace(self.input_shape) for k in self._feature_names},
            }
        )
        self._normalizers = [self._norms_dict[k] for k in self._feature_names]

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
