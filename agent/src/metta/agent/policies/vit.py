import logging
from typing import List

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.lstm import LSTMConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsLatentAttnConfig, ObsPerceiverLatentConfig, ObsSelfAttnConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.components.obs_trim import ObsTokenTrimConfig
from metta.agent.policy import PolicyArchitecture

logger = logging.getLogger(__name__)


class ViTSmallConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _hidden_size = 128
    _embedding_dim = 16

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens"),
        ObsAttrEmbedFourierConfig(in_key="obs_shim_tokens", out_key="obs_attr_embed_fourier"),
        ObsLatentAttnConfig(in_key="obs_attr_embed_fourier", out_key="obs_latent_attn", feat_dim=37, out_dim=48),
        ObsSelfAttnConfig(in_key="obs_latent_attn", out_key="obs_self_attn", feat_dim=48, out_dim=_hidden_size),
        LSTMConfig(
            in_key="obs_self_attn",
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
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        ActorQueryConfig(
            in_key="core",
            out_key="actor_query",
            hidden_size=_hidden_size,
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


class ViTDefaultConfig(PolicyArchitecture):
    """Speed-optimized ViT variant with lighter token embeddings and attention stack."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _hidden_size = 128
    _embedding_dim = 16

    _token_embed_dim = 8
    _fourier_freqs = 6
    _latent_dim = 48
    _lstm_latent = 96
    _critic_hidden = 128

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens"),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsTokenTrimConfig(
            in_key="obs_attr_embed",
            out_key="obs_attr_trimmed",
            max_tokens=None,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_trimmed",
            out_key="obs_latent_attn",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=16,
            num_heads=4,
            num_layers=2,
        ),
        LSTMConfig(
            in_key="obs_latent_attn",
            out_key="core",
            latent_size=_latent_dim,
            hidden_size=_lstm_latent,
            num_layers=1,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_lstm_latent,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        ActorQueryConfig(
            in_key="core",
            out_key="actor_query",
            hidden_size=_lstm_latent,
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
