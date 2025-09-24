from typing import List

from pydantic import Field

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig
from metta.agent.components.agalite_transformer import AGaLiTeTransformerConfig
from metta.agent.components.cnn_encoder import CNNEncoderConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_shim import ObsShimBoxConfig
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
) -> List[ComponentConfig]:
    kernel_cfg = kernel or AGaLiTeKernelConfig()
    eta_value = eta
    if kernel_cfg.name in {"pp_relu", "pp_eluplus1", "dpfp"}:
        eta_value = max(kernel_cfg.nu, 1)

    return [
        ObsShimBoxConfig(in_key="env_obs", out_key="obs_box"),
        CNNEncoderConfig(
            in_key="obs_box",
            out_key="encoded_obs",
            cnn1_cfg={"out_channels": 64, "kernel_size": 5, "stride": 3},
            cnn2_cfg={"out_channels": 64, "kernel_size": 3, "stride": 1},
            fc1_cfg={"out_features": 128},
            encoded_obs_cfg={"out_features": hidden_size},
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

    components: List[ComponentConfig] = Field(
        default_factory=lambda: _build_components(
            hidden_size=192,
            embedding_dim=16,
            n_layers=2,
            n_heads=4,
            feedforward_size=768,
            eta=4,
            r=8,
            mode="agalite",
            dropout=0.0,
            kernel=AGaLiTeKernelConfig(name="relu", nu=4),
        )
    )

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class AGaLiTeImprovedConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    components: List[ComponentConfig] = Field(
        default_factory=lambda: _build_components(
            hidden_size=256,
            embedding_dim=16,
            n_layers=4,
            n_heads=8,
            feedforward_size=1024,
            eta=4,
            r=16,
            mode="agalite",
            dropout=0.1,
            kernel=AGaLiTeKernelConfig(name="eluplus1", nu=4),
        )
    )

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")
