from typing import List

from cortex.stacks import build_cortex_auto_config

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy_architecture import PolicyArchitecture


class ViTSize2Config(PolicyArchitecture):
    """Slightly larger than default ViT policy."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 256
    _actor_hidden = 512

    # Whether training passes cached pre-state to the Cortex core
    pass_state_during_training: bool = False
    _critic_hidden = 512

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
        CortexTDConfig(
            in_key="obs_latent_attn",
            out_key="core",
            d_hidden=_latent_dim,
            out_features=_latent_dim,
            key_prefix="vit_cortex_state",
            stack_cfg=build_cortex_auto_config(
                d_hidden=_latent_dim,
                num_layers=2,
                pattern="L",
                post_norm=False,
            ),
            pass_state_during_training=pass_state_during_training,
        ),
        MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_latent_dim,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_latent_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        MLPConfig(
            in_key="core",
            out_key="h_values",
            name="gtd_aux",
            in_features=_latent_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=_actor_hidden),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")
