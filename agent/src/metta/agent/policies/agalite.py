from typing import List

from pydantic import Field

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig
from metta.agent.components.agalite_transformer import AGaLiTeTransformerConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture


def _build_components(
    *,
    hidden_size: int,
    embedding_dim: int,
    n_layers: int,
    n_heads: int,
    feedforward_size: int,
    eta: int,
    r: int,
    dropout: float,
    kernel: AGaLiTeKernelConfig | None = None,
    max_tokens: int = 64,
    attr_embed_dim: int = 12,
    fourier_freqs: int = 4,
    num_latents: int = 16,
    critic_hidden: List[int] | None = None,
    actor_hidden_dim: int | None = None,
    reset_on_terminate: bool = True,
) -> List[ComponentConfig]:
    kernel_cfg = kernel or AGaLiTeKernelConfig()
    eta_value = eta
    if kernel_cfg.name in {"pp_relu", "pp_eluplus1", "dpfp"}:
        eta_value = max(kernel_cfg.nu, 1)

    feat_dim = attr_embed_dim + (4 * fourier_freqs) + 1
    critic_hidden_features = critic_hidden or [1024]
    actor_hidden = actor_hidden_dim or hidden_size
    actor_out_key = "actor_features"

    return [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=max_tokens),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=attr_embed_dim,
            num_freqs=fourier_freqs,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="encoded_obs",
            feat_dim=feat_dim,
            latent_dim=hidden_size,
            num_latents=num_latents,
            num_heads=n_heads,
            num_layers=1,
            pool="mean",
        ),
        AGaLiTeTransformerConfig(
            in_key="encoded_obs",
            out_key="core",
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            feedforward_size=feedforward_size,
            eta=eta_value,
            r=r,
            dropout=dropout,
            kernel=kernel_cfg,
            reset_on_terminate=reset_on_terminate,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=hidden_size,
            hidden_features=critic_hidden_features,
            out_features=1,
            nonlinearity="ReLU",
            output_nonlinearity=None,
        ),
        MLPConfig(
            in_key="core",
            out_key=actor_out_key,
            name="actor_mlp",
            in_features=hidden_size,
            hidden_features=[actor_hidden],
            out_features=actor_hidden,
            nonlinearity="ReLU",
            output_nonlinearity=None,
        ),
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=embedding_dim),
        ActorQueryConfig(
            in_key=actor_out_key,
            out_key="actor_query",
            hidden_size=actor_hidden,
            embed_dim=embedding_dim,
        ),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=embedding_dim,
            hidden_size=actor_hidden,
        ),
    ]


class AGaLiTeConfig(PolicyArchitecture):
    """AGaLiTe configuration aligned with the published architecture."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"
    learning_rate_hint: float = 1e-3

    components: List[ComponentConfig] = Field(
        default_factory=lambda: _build_components(
            hidden_size=64,
            embedding_dim=16,
            n_layers=3,
            n_heads=4,
            feedforward_size=256,
            eta=6,
            r=2,
            dropout=0.05,
            kernel=AGaLiTeKernelConfig(name="eluplus1", nu=4),
            max_tokens=64,
            attr_embed_dim=12,
            fourier_freqs=4,
            num_latents=16,
            critic_hidden=[256],
            actor_hidden_dim=128,
            reset_on_terminate=False,
        )
    )

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def policy_defaults(self) -> dict[str, object]:
        return {"learning_rate_hint": self.learning_rate_hint}


__all__ = ["AGaLiTeConfig"]
