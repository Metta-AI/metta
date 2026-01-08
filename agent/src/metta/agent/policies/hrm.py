from typing import List

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.hrm import HRMReasoningConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture


class HRMPolicyConfig(PolicyArchitecture):
    """Component-based Hierarchical Reasoning Model (HRM) policy configuration."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 256
    _token_embed_dim = 16
    _fourier_freqs = 3
    _embedding_dim = 16
    _actor_hidden = 512
    _critic_hidden = 1024

    components: List[ComponentConfig] = [
        # Token-based observation pipeline
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="hrm_obs_encoded",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_embed_dim,
            num_latents=16,
            num_heads=4,
            num_layers=2,
        ),
        # Hierarchical reasoning
        HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="core",
            embed_dim=_embed_dim,
            num_layers=4,
            num_heads=8,
            ffn_expansion=4.0,
        ),
        # Actor and Critic using ViT-style components
        MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_embed_dim,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_embed_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        MLPConfig(
            in_key="core",
            out_key="h_values",
            name="gtd_aux",
            in_features=_embed_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=_actor_hidden),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class HRMTinyConfig(PolicyArchitecture):
    """Tiny version of HRM for testing and low-memory environments."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 64
    _token_embed_dim = 8
    _fourier_freqs = 3
    _embedding_dim = 16
    _actor_hidden = 256
    _critic_hidden = 512

    components: List[ComponentConfig] = [
        # Token-based observation pipeline (minimal config)
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="hrm_obs_encoded",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_embed_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        # Hierarchical reasoning (minimal layers, smaller FFN)
        HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="core",
            embed_dim=_embed_dim,
            num_layers=1,
            num_heads=2,
            ffn_expansion=2.0,
        ),
        # Actor and Critic using ViT-style components
        MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_embed_dim,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_embed_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        MLPConfig(
            in_key="core",
            out_key="h_values",
            name="gtd_aux",
            in_features=_embed_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=_actor_hidden),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")
