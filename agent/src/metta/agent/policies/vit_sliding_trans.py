import logging
import typing

import metta.agent.components.actor
import metta.agent.components.component_config
import metta.agent.components.misc
import metta.agent.components.obs_enc
import metta.agent.components.obs_shim
import metta.agent.components.obs_tokenizers
import metta.agent.policies.sliding_transformer
import metta.agent.policy

logger = logging.getLogger(__name__)


class ViTSlidingTransConfig(metta.agent.policy.PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 64
    _token_embed_dim = 8
    _fourier_freqs = 3
    _core_out_dim = 32
    _memory_num_layers = 2

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
            out_key="encoded_obs",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        metta.agent.policies.sliding_transformer.SlidingTransformerConfig(
            in_key="encoded_obs",
            out_key="core",
            hidden_size=_core_out_dim,
            latent_size=_latent_dim,
            num_layers=_memory_num_layers,
        ),
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[1024],
        ),
        metta.agent.components.actor.ActorHeadConfig(in_key="core", out_key="logits", input_dim=_core_out_dim),
    ]

    action_probs_config: metta.agent.components.actor.ActionProbsConfig = (
        metta.agent.components.actor.ActionProbsConfig(in_key="logits")
    )
