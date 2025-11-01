from typing import List

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.meta_cog.mc_action import MCActionEmbeddingConfig
from metta.agent.meta_cog.mc_actor import (
    MCActionProbsConfig,
    MCActorKeyConfig,
    MCActorQueryConfig,
)
from metta.agent.meta_cog.mc_lstm_reset import MCLSTMResetConfig
from metta.agent.meta_cog.mc_obs_shim_tokens import MCObsShimTokensConfig
from metta.agent.policy import PolicyArchitecture


class MCViTResetConfig(PolicyArchitecture):
    """Speed-optimized ViT variant with lighter token embeddings and attention stack."""

    class_path: str = "metta.agent.meta_cog.mc_policy_auto_builder.MCPolicyAutoBuilder"

    think_first: bool = True

    _embedding_dim = 16

    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 64
    _lstm_latent = 32
    _actor_hidden = 256
    _mc_actor_hidden = 256
    _critic_hidden = 512
    _mc_embedding_dim = 16

    components: List[ComponentConfig] = [
        MCObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="obs_latent_attn",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        MCLSTMResetConfig(
            in_key="obs_latent_attn",
            out_key="core",
            latent_size=_latent_dim,
            hidden_size=_lstm_latent,
            num_layers=1,
        ),
        MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_lstm_latent,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_lstm_latent,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim, name="actor_embeds"),
        ActorQueryConfig(
            in_key="actor_hidden",
            out_key="actor_query",
            name="actor_query",
            hidden_size=_actor_hidden,
            embed_dim=_embedding_dim,
        ),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            name="actor_key",
            embed_dim=_embedding_dim,
        ),
        MLPConfig(
            in_key="core",
            out_key="mc_actor_hidden",
            name="mc_actor_mlp",
            in_features=_lstm_latent,
            hidden_features=[_mc_actor_hidden],
            out_features=_mc_actor_hidden,
        ),
        MCActionEmbeddingConfig(out_key="mc_action_embedding", embedding_dim=_mc_embedding_dim, name="mc_actor_embeds"),
        MCActorQueryConfig(
            in_key="mc_actor_hidden",
            out_key="mc_actor_query",
            name="mc_actor_query",
            hidden_size=_mc_actor_hidden,
            embed_dim=_mc_embedding_dim,
        ),
        MCActorKeyConfig(
            query_key="mc_actor_query",
            embedding_key="mc_action_embedding",
            out_key="mc_logits",
            name="mc_actor_key",
            embed_dim=_mc_embedding_dim,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits", name="action_probs")
    mc_action_probs_config: MCActionProbsConfig = MCActionProbsConfig(in_key="mc_logits", name="mc_action_probs")
