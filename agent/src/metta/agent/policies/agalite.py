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
    mode: str,
    dropout: float,
    kernel: AGaLiTeKernelConfig | None = None,
    max_tokens: int = 64,
    attr_embed_dim: int = 12,
    fourier_freqs: int = 4,
    num_latents: int = 16,
) -> List[ComponentConfig]:
    kernel_cfg = kernel or AGaLiTeKernelConfig()
    eta_value = eta
    if kernel_cfg.name in {"pp_relu", "pp_eluplus1", "dpfp"}:
        eta_value = max(kernel_cfg.nu, 1)

    feat_dim = attr_embed_dim + (4 * fourier_freqs) + 1

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
            num_layers=2,
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
            mode=mode,
            dropout=dropout,
            kernel=kernel_cfg,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=hidden_size,
            hidden_features=[1024],
            out_features=1,
            nonlinearity="ReLU",
            output_nonlinearity=None,
        ),
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=embedding_dim),
        ActorQueryConfig(
            in_key="core",
            out_key="actor_query",
            hidden_size=hidden_size,
            embed_dim=embedding_dim,
        ),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=embedding_dim,
        ),
    ]


class AGaLiTeConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    # Parameter budget tuned to match Transformer GTrXL/TRXL cores (~14k params).
    components: List[ComponentConfig] = Field(
        default_factory=lambda: _build_components(
            hidden_size=24,
            embedding_dim=16,
            n_layers=1,
            n_heads=2,
            feedforward_size=48,
            eta=2,
            r=4,
            mode="agalite",
            dropout=0.05,
            kernel=AGaLiTeKernelConfig(name="relu", nu=2),
            max_tokens=48,
            attr_embed_dim=8,
            fourier_freqs=3,
            num_latents=12,
        )
    )

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class AGaLiTeImprovedConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    # Parameter budget tuned to match Transformer TRXL_NVIDIA core (~61k params).
    components: List[ComponentConfig] = Field(
        default_factory=lambda: _build_components(
            hidden_size=32,
            embedding_dim=16,
            n_layers=2,
            n_heads=4,
            feedforward_size=160,
            eta=2,
            r=4,
            mode="agalite",
            dropout=0.05,
            kernel=AGaLiTeKernelConfig(name="eluplus1", nu=2),
            max_tokens=48,
            attr_embed_dim=8,
            fourier_freqs=3,
            num_latents=12,
        )
    )

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")
