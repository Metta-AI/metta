import typing

import metta.agent.components.action
import metta.agent.components.actor
import metta.agent.components.component_config
import metta.agent.components.drama
import metta.agent.components.misc
import metta.agent.components.obs_enc
import metta.agent.components.obs_shim
import metta.agent.components.obs_tokenizers
import metta.agent.policy


class DramaPolicyConfig(metta.agent.policy.PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 48
    _token_embed_dim = 4
    _fourier_freqs = 2
    _embed_dim = 12
    _core_out_dim = 96

    components: typing.List[metta.agent.components.component_config.ComponentConfig] = [
        metta.agent.components.obs_shim.ObsShimTokensConfig(in_key="env_obs", out_key="obs_tokens", max_tokens=48),
        metta.agent.components.obs_tokenizers.ObsAttrEmbedFourierConfig(
            in_key="obs_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        metta.agent.components.obs_enc.ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="encoded_obs",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=8,
            num_heads=2,
            num_layers=1,
        ),
        metta.agent.components.drama.DramaWorldModelConfig(
            in_key="encoded_obs",
            out_key="core",
            action_key="last_actions",
            stoch_dim=_latent_dim,
            d_model=_core_out_dim,
            d_intermediate=_core_out_dim * 2,
            n_layer=1,
        ),
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[192],
        ),
        metta.agent.components.action.ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embed_dim),
        metta.agent.components.actor.ActorQueryConfig(
            in_key="core", out_key="actor_query", hidden_size=_core_out_dim, embed_dim=_embed_dim
        ),
        metta.agent.components.actor.ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embed_dim,
        ),
    ]

    action_probs_config: metta.agent.components.actor.ActionProbsConfig = (
        metta.agent.components.actor.ActionProbsConfig(in_key="logits")
    )


__all__ = ["DramaPolicyConfig"]
