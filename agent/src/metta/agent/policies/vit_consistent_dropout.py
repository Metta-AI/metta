from typing import List

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.lstm import LSTMConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.misc_consistent_dropout import MLPConsistentDropoutConfig
from metta.agent.components.obs_enc_consistent_dropout import ObsPerceiverLatentConsistentDropoutConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture


class ViTConsistentDropoutConfig(PolicyArchitecture):
    """ViT variant with consistent dropout for stable RL policy gradients.

    This architecture uses consistent dropout which ensures the same dropout
    mask is reused during both rollout and gradient computation, preventing
    bias in policy gradients as described in Hausknecht & Wagener's work.

    The consistent dropout is applied in:
    - Actor MLP hidden layers
    - ObsPerceiverLatent attention and MLP blocks

    The critic path uses standard dropout since value function estimation
    doesn't require the same mask consistency.
    """

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 64
    _lstm_latent = 32
    _actor_hidden = 256
    _critic_hidden = 512
    _dropout_p = 0.2

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsPerceiverLatentConsistentDropoutConfig(
            in_key="obs_attr_embed",
            out_key="obs_latent_attn",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
            dropout_p=_dropout_p,
        ),
        LSTMConfig(
            in_key="obs_latent_attn",
            out_key="core",
            latent_size=_latent_dim,
            hidden_size=_lstm_latent,
            num_layers=1,
        ),
        MLPConsistentDropoutConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_lstm_latent,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
            dropout_p=_dropout_p,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_lstm_latent,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=_actor_hidden),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")
