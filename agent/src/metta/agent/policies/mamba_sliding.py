from typing import List

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.mamba.config import MambaBackboneConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture


class MambaSlidingConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 64
    _token_embed_dim = 8
    _fourier_freqs = 3
    _embed_dim = 16
    _core_out_dim = 128

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
            out_key="encoded_obs",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        MambaBackboneConfig(
            in_key="encoded_obs",
            out_key="core",
            input_dim=_latent_dim,
            d_model=_core_out_dim,
            d_intermediate=_core_out_dim * 2,
            n_layer=2,
            max_cache_size=128,
            pool="mean",
            ssm_cfg={"layer": "Mamba2"},
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[256],
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


def _update_ssm_layer(config: MambaSlidingConfig, layer: str) -> MambaSlidingConfig:
    if layer != "Mamba2":
        raise ValueError(f"Unsupported SSM layer '{layer}'. Only 'Mamba2' is available.")
    for component in config.components:
        if isinstance(component, MambaBackboneConfig):
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
