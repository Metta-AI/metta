import typing

import metta.agent.components.action
import metta.agent.components.actor
import metta.agent.components.component_config
import metta.agent.components.hrm
import metta.agent.components.misc
import metta.agent.components.obs_enc
import metta.agent.components.obs_shim
import metta.agent.components.obs_tokenizers
import metta.agent.policy


class HRMPolicyConfig(metta.agent.policy.PolicyArchitecture):
    """Component-based Hierarchical Reasoning Model (HRM) policy configuration."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 256
    _token_embed_dim = 16
    _fourier_freqs = 3
    _embedding_dim = 16
    _actor_hidden = 512
    _critic_hidden = 1024

    components: typing.List[metta.agent.components.component_config.ComponentConfig] = [
        # Token-based observation pipeline
        metta.agent.components.obs_shim.ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        metta.agent.components.obs_tokenizers.ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        metta.agent.components.obs_enc.ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="hrm_obs_encoded",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_embed_dim,
            num_latents=16,
            num_heads=4,
            num_layers=2,
        ),
        # Hierarchical reasoning
        metta.agent.components.hrm.HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="core",
            embed_dim=_embed_dim,
            num_layers=4,
            num_heads=8,
            ffn_expansion=4.0,
        ),
        # Actor and Critic using ViT-style components
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_embed_dim,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_embed_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        metta.agent.components.action.ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        metta.agent.components.actor.ActorQueryConfig(
            in_key="actor_hidden",
            out_key="actor_query",
            hidden_size=_actor_hidden,
            embed_dim=_embedding_dim,
        ),
        metta.agent.components.actor.ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embedding_dim,
        ),
    ]

    action_probs_config: metta.agent.components.actor.ActionProbsConfig = (
        metta.agent.components.actor.ActionProbsConfig(in_key="logits")
    )


class HRMTinyConfig(metta.agent.policy.PolicyArchitecture):
    """Tiny version of HRM for testing and low-memory environments."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 64
    _token_embed_dim = 8
    _fourier_freqs = 3
    _embedding_dim = 16
    _actor_hidden = 256
    _critic_hidden = 512

    components: typing.List[metta.agent.components.component_config.ComponentConfig] = [
        # Token-based observation pipeline (minimal config)
        metta.agent.components.obs_shim.ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        metta.agent.components.obs_tokenizers.ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        metta.agent.components.obs_enc.ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="hrm_obs_encoded",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_embed_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        # Hierarchical reasoning (minimal layers, smaller FFN)
        metta.agent.components.hrm.HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="core",
            embed_dim=_embed_dim,
            num_layers=1,
            num_heads=2,
            ffn_expansion=2.0,
        ),
        # Actor and Critic using ViT-style components
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_embed_dim,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_embed_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        metta.agent.components.action.ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        metta.agent.components.actor.ActorQueryConfig(
            in_key="actor_hidden",
            out_key="actor_query",
            hidden_size=_actor_hidden,
            embed_dim=_embedding_dim,
        ),
        metta.agent.components.actor.ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embedding_dim,
        ),
    ]

    action_probs_config: metta.agent.components.actor.ActionProbsConfig = (
        metta.agent.components.actor.ActionProbsConfig(in_key="logits")
    )
