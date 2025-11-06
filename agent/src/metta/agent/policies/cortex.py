import typing

import cortex.stacks

import metta.agent.components.action
import metta.agent.components.actor
import metta.agent.components.component_config
import metta.agent.components.cortex
import metta.agent.components.misc
import metta.agent.components.obs_enc
import metta.agent.components.obs_shim
import metta.agent.components.obs_tokenizers
import metta.agent.policy


class CortexBaseConfig(metta.agent.policy.PolicyArchitecture):
    """ViT-style policy with Cortex stack (xLSTM) replacing LSTM core."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embedding_dim = 16

    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 64
    _core_out = 32  # align with ViTReset LSTM latent
    _actor_hidden = 256
    _critic_hidden = 512

    components: typing.List[metta.agent.components.component_config.ComponentConfig] = [
        metta.agent.components.obs_shim.ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        metta.agent.components.obs_tokenizers.ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        metta.agent.components.obs_enc.ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="obs_latent_attn",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        metta.agent.components.cortex.CortexTDConfig(
            in_key="obs_latent_attn",
            out_key="core",
            d_hidden=_latent_dim,
            out_features=_core_out,
            # Default to the mixed Cortex auto stack (Axon/mLSTM/sLSTM) via config.
            stack_cfg=cortex.stacks.build_cortex_auto_config(d_hidden=_latent_dim, post_norm=True),
            key_prefix="cortex_state",
        ),
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_core_out,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out,
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
