from typing import List

from cortex.stacks.xlstm import build_xlstm_stack

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture


class CortexBaseConfig(PolicyArchitecture):
    """ViT-style policy that uses a Cortex stack as the core memory module.

    Matches the ViTReset layout but swaps the LSTM core for a Cortex stack
    (defaults to xLSTM). Keeps actor/critic heads and dims consistent by
    projecting the Cortex output to the same hidden size as the original LSTM.
    """

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embedding_dim = 16

    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 64
    _core_out = 32  # align with ViTReset LSTM latent
    _actor_hidden = 256
    _critic_hidden = 512
    _flash_window = 48
    _flash_dropout = 0.0

    components: List["PolicyArchitecture"] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
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
        CortexTDConfig(
            in_key="obs_latent_attn",
            out_key="core",
            d_hidden=_latent_dim,
            out_features=_core_out,
            stack=build_xlstm_stack(
                d_hidden=_latent_dim,
                num_blocks=2,
                mlstm_num_heads=2,
                slstm_num_heads=2,
                mlstm_proj_factor=2.0,
                slstm_proj_factor=1.5,
                mlstm_chunk_size=128,
                conv1d_kernel_size=4,
                dropout=0.0,
                post_norm=True,
                flash_window_size=_flash_window,
                flash_num_heads=4,
                flash_dropout=_flash_dropout,
            ),
            key_prefix="cortex_state",
        ),
        MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_core_out,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out,
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
