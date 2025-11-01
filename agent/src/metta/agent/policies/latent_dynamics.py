"""Policy architecture with latent-variable dynamics model for model-based RL.

This policy integrates a latent dynamics model that learns to predict future states
by encoding transitions into stochastic latent variables, following the approach from
"Learning Dynamics Model in Reinforcement Learning by Incorporating the Long Term Future"
(Ke et al., ICLR 2019).
"""

import logging
from typing import List

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.dynamics import LatentDynamicsConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policies.sliding_transformer import SlidingTransformerConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.training import EnvironmentMetaData
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)


class LatentDynamicsPolicyConfig(PolicyArchitecture):
    """Policy with latent-variable dynamics model.

    Architecture:
    1. Observation tokenization and encoding
    2. Perceiver for compression
    3. Sliding transformer for temporal modeling
    4. Latent dynamics model for learning state transitions
    5. Actor and critic heads
    """

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 64
    _token_embed_dim = 8
    _fourier_freqs = 3
    _core_out_dim = 32
    _memory_num_layers = 2

    # Dynamics model hyperparameters
    _dynamics_latent_dim = 32
    _dynamics_encoder_hidden = [64, 64]
    _dynamics_decoder_hidden = [64, 64]
    _dynamics_auxiliary_hidden = [32]

    components: List[ComponentConfig] = [
        # Observation processing
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="encoded_obs",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        # Temporal modeling
        SlidingTransformerConfig(
            in_key="encoded_obs",
            out_key="core",
            hidden_size=_core_out_dim,
            latent_size=_latent_dim,
            num_layers=_memory_num_layers,
        ),
        # Latent dynamics model
        LatentDynamicsConfig(
            name="latent_dynamics",
            in_key="encoded_obs",
            out_key="dynamics_latent",
            action_key="last_actions",
            latent_dim=_dynamics_latent_dim,
            encoder_hidden=_dynamics_encoder_hidden,
            decoder_hidden=_dynamics_decoder_hidden,
            auxiliary_hidden=_dynamics_auxiliary_hidden,
            beta_kl=0.01,
            gamma_auxiliary=1.0,
            future_horizon=5,
            future_type="returns",
            use_auxiliary=True,
        ),
        # Critic
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[1024],
        ),
        # Actor
        ActorHeadConfig(in_key="core", out_key="logits", input_dim=_core_out_dim),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, env_metadata: EnvironmentMetaData) -> Policy:
        """Create policy instance."""
        AgentClass = load_symbol(self.class_path)
        policy = AgentClass(env_metadata, self)
        return policy


class LatentDynamicsTinyConfig(PolicyArchitecture):
    """Tiny version of latent dynamics policy for fast testing."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 32
    _token_embed_dim = 4
    _fourier_freqs = 2
    _core_out_dim = 16
    _memory_num_layers = 1

    # Smaller dynamics model
    _dynamics_latent_dim = 16
    _dynamics_encoder_hidden = [32]
    _dynamics_decoder_hidden = [32]
    _dynamics_auxiliary_hidden = [16]

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=32),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="encoded_obs",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=8,
            num_heads=2,
            num_layers=1,
        ),
        SlidingTransformerConfig(
            in_key="encoded_obs",
            out_key="core",
            hidden_size=_core_out_dim,
            latent_size=_latent_dim,
            num_layers=_memory_num_layers,
        ),
        LatentDynamicsConfig(
            name="latent_dynamics",
            in_key="encoded_obs",
            out_key="dynamics_latent",
            action_key="last_actions",
            latent_dim=_dynamics_latent_dim,
            encoder_hidden=_dynamics_encoder_hidden,
            decoder_hidden=_dynamics_decoder_hidden,
            auxiliary_hidden=_dynamics_auxiliary_hidden,
            beta_kl=0.005,
            gamma_auxiliary=0.5,
            future_horizon=3,
            future_type="returns",
            use_auxiliary=True,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[256],
        ),
        ActorHeadConfig(in_key="core", out_key="logits", input_dim=_core_out_dim),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, env_metadata: EnvironmentMetaData) -> Policy:
        """Create policy instance."""
        AgentClass = load_symbol(self.class_path)
        policy = AgentClass(env_metadata, self)
        return policy
