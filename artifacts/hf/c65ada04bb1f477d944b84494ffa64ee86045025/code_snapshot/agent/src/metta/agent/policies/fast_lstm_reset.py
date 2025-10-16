import logging
from typing import List

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.cnn_encoder import CNNEncoderConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.lstm_reset import LSTMResetConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_shim import ObsShimBoxConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.training import EnvironmentMetaData
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)


class FastLSTMResetConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _hidden_size = 128
    components: List[ComponentConfig] = [
        ObsShimBoxConfig(in_key="env_obs", out_key="obs_shim_box"),
        CNNEncoderConfig(in_key="obs_shim_box", out_key="encoded_obs"),
        LSTMResetConfig(
            in_key="encoded_obs",
            out_key="core",
            latent_size=_hidden_size,
            hidden_size=_hidden_size,
            num_layers=2,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_hidden_size,
            out_features=1,
            hidden_features=[1024],
        ),
        ActorHeadConfig(in_key="core", out_key="logits", input_dim=_hidden_size),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, env_metadata: EnvironmentMetaData) -> Policy:
        AgentClass = load_symbol(self.class_path)
        policy = AgentClass(env_metadata, self)
        policy.burn_in_steps = 128  # async factor of 2 * bptt of 64 although this isn't necessarily a function of bptt

        return policy
