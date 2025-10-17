from typing import List

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.lstm import LSTMConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture

class ViTGRPOConfig(PolicyArchitecture):
    """ViT architecture optimized for GRPO - no critic network.

    This architecture is identical to ViTDefaultConfig but without the value/critic
    network, making it faster for GRPO training which doesn't need value predictions.
    """

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embedding_dim = 16
    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 64
    _lstm_latent = 32
    _actor_hidden = 256

    components: List[ComponentConfig] = [
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
        LSTMConfig(
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
        ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=_actor_hidden),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")
