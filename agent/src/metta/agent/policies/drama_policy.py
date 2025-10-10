from typing import List

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import (
    ActionProbsConfig,
    ActorKeyConfig,
    ActorQueryConfig,
)
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.drama import DramaWorldModelConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture


class DramaPolicyConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 48
    _token_embed_dim = 4
    _fourier_freqs = 2
    _embed_dim = 12
    _core_out_dim = 96

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_tokens", max_tokens=32),
        ObsAttrEmbedFourierConfig(
            in_key="obs_tokens",
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
        DramaWorldModelConfig(
            in_key="encoded_obs",
            out_key="core",
            action_key="last_actions",
            stoch_dim=_latent_dim,
            d_model=_core_out_dim,
            d_intermediate=_core_out_dim * 2,
            n_layer=1,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[192],
        ),
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embed_dim),
        ActorQueryConfig(in_key="core", out_key="actor_query", hidden_size=_core_out_dim, embed_dim=_embed_dim),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embed_dim,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


__all__ = ["DramaPolicyConfig"]
