import typing

import metta.agent.components.action
import metta.agent.components.actor
import metta.agent.components.component_config
import metta.agent.components.mamba.config
import metta.agent.components.misc
import metta.agent.components.obs_enc
import metta.agent.components.obs_shim
import metta.agent.components.obs_tokenizers
import metta.agent.policy


class MambaSlidingConfig(metta.agent.policy.PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 32
    _token_embed_dim = 8
    _fourier_freqs = 3
    _embed_dim = 16
    _core_out_dim = 32

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
            num_layers=1,
        ),
        metta.agent.components.mamba.config.MambaBackboneConfig(
            in_key="encoded_obs",
            out_key="core",
            input_dim=_latent_dim,
            d_model=_core_out_dim,
            d_intermediate=_core_out_dim * 4,
            n_layer=1,
            max_cache_size=192,
            pool="mean",
            ssm_expand=2,
            ssm_headdim=16,
            use_mem_eff_path=True,
        ),
        metta.agent.components.misc.MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[128],
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


def _update_ssm_layer(config: MambaSlidingConfig, layer: str) -> MambaSlidingConfig:
    if layer != "Mamba2":
        raise ValueError(f"Unsupported SSM layer '{layer}'. Only 'Mamba2' is available.")
    for component in config.components:
        if isinstance(component, metta.agent.components.mamba.config.MambaBackboneConfig):
            next_cfg = dict(component.ssm_cfg) if component.ssm_cfg else {}
            next_cfg["layer"] = layer
            component.ssm_cfg = next_cfg
    return config


def mamba_policy_config() -> MambaSlidingConfig:
    """Return a default Mamba sliding policy configuration."""

    return MambaSlidingConfig()


def mamba2_policy_config() -> MambaSlidingConfig:
    """Return a Mamba sliding policy that selects the Mamba2 mixer."""

    return _update_ssm_layer(MambaSlidingConfig(), layer="Mamba2")


__all__ = ["MambaSlidingConfig", "mamba_policy_config", "mamba2_policy_config"]
