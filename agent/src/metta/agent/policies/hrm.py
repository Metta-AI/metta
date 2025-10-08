from typing import List

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.hrm import HRMReasoningConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture
from metta.agent.policy_auto_builder import PolicyAutoBuilder


class HRMPolicy(PolicyAutoBuilder):
    """Component-based Hierarchical Reasoning Model (HRM) policy implementation."""

    def __init__(self, env_metadata):
        _embed_dim = 256
        _token_embed_dim = 16
        _fourier_freqs = 3
        _embedding_dim = 16
        _actor_hidden = 512
        _critic_hidden = 1024

        config = PolicyArchitecture(
            class_path="metta.agent.policy_auto_builder.PolicyAutoBuilder",
            components=[
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
                ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
                ActorQueryConfig(
                    in_key="actor_hidden",
                    out_key="actor_query",
                    hidden_size=_actor_hidden,
                    embed_dim=_embedding_dim,
                ),
                ActorKeyConfig(
                    query_key="actor_query",
                    embedding_key="action_embedding",
                    out_key="logits",
                    embed_dim=_embedding_dim,
                ),
            ],
            action_probs_config=ActionProbsConfig(in_key="logits"),
        )

        super().__init__(env_metadata, config)


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
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        ActorQueryConfig(
            in_key="actor_hidden",
            out_key="actor_query",
            hidden_size=_actor_hidden,
            embed_dim=_embedding_dim,
        ),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embedding_dim,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class HRMCompactConfig(PolicyArchitecture):
    """Compact version of HRM with smaller dimensions for faster training."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 128
    _token_embed_dim = 12
    _fourier_freqs = 2
    _embedding_dim = 16
    _actor_hidden = 384
    _critic_hidden = 768

    components: List[ComponentConfig] = [
        # Token-based observation pipeline (reduced size)
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=40),
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
            num_layers=1,
        ),
        # Hierarchical reasoning (fewer layers and heads)
        HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="core",
            embed_dim=_embed_dim,
            num_layers=2,
            num_heads=4,
            ffn_expansion=3.0,
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
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        ActorQueryConfig(
            in_key="actor_hidden",
            out_key="actor_query",
            hidden_size=_actor_hidden,
            embed_dim=_embedding_dim,
        ),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embedding_dim,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class HRMLargeConfig(PolicyArchitecture):
    """Large version of HRM with more capacity for complex reasoning."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 512
    _token_embed_dim = 24
    _fourier_freqs = 4
    _embedding_dim = 24
    _actor_hidden = 1024
    _critic_hidden = 2048

    components: List[ComponentConfig] = [
        # Token-based observation pipeline (larger capacity)
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=64),
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
            num_latents=24,
            num_heads=8,
            num_layers=3,
        ),
        # Hierarchical reasoning (more layers and heads)
        HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="core",
            embed_dim=_embed_dim,
            num_layers=8,
            num_heads=16,
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
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        ActorQueryConfig(
            in_key="actor_hidden",
            out_key="actor_query",
            hidden_size=_actor_hidden,
            embed_dim=_embedding_dim,
        ),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embedding_dim,
        ),
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
            track_gradients=True,  # Enable gradient tracking for monitoring
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
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        ActorQueryConfig(
            in_key="actor_hidden",
            out_key="actor_query",
            hidden_size=_actor_hidden,
            embed_dim=_embedding_dim,
        ),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embedding_dim,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


# Default HRM configuration
HRMDefaultConfig = HRMPolicyConfig
