import logging
import typing

import metta.agent.components.actor
import metta.agent.components.cnn_encoder
import metta.agent.components.component_config
import metta.agent.components.lstm_reset
import metta.agent.components.misc
import metta.agent.components.obs_shim
import metta.agent.policy
import mettagrid.policy.policy_env_interface
import mettagrid.util.module

logger = logging.getLogger(__name__)


class FastLSTMResetConfig(metta.agent.policy.PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _hidden_size = 128
    components: typing.List[metta.agent.components.component_config.ComponentConfig] = [
        metta.agent.components.obs_shim.ObsShimBoxConfig(in_key="env_obs", out_key="obs_shim_box"),
        metta.agent.components.cnn_encoder.CNNEncoderConfig(in_key="obs_shim_box", out_key="encoded_obs"),
        metta.agent.components.lstm_reset.LSTMResetConfig(
            in_key="encoded_obs",
            out_key="core",
            latent_size=_hidden_size,
            hidden_size=_hidden_size,
            num_layers=2,
        ),
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_hidden_size,
            out_features=1,
            hidden_features=[1024],
        ),
        metta.agent.components.actor.ActorHeadConfig(in_key="core", out_key="logits", input_dim=_hidden_size),
    ]

    action_probs_config: metta.agent.components.actor.ActionProbsConfig = (
        metta.agent.components.actor.ActionProbsConfig(in_key="logits")
    )

    def make_policy(
        self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface
    ) -> metta.agent.policy.Policy:
        AgentClass = mettagrid.util.module.load_symbol(self.class_path)
        policy = AgentClass(policy_env_info, self)
        policy.burn_in_steps = 128  # async factor of 2 * bptt of 64 although this isn't necessarily a function of bptt

        return policy
