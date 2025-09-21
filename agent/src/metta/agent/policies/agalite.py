from typing import List

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.agalite_transformer import AGaLiTeTransformerConfig
from metta.agent.components.cnn_encoder import CNNEncoderConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_shim import ObsShimBoxConfig
from metta.agent.policy import PolicyArchitecture


class AGaLiTeConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _hidden_size = 192
    _embedding_dim = 16

    components: List[ComponentConfig] = [
        ObsShimBoxConfig(in_key="env_obs", out_key="obs_box"),
        CNNEncoderConfig(
            in_key="obs_box",
            out_key="encoded_obs",
            cnn1_cfg={"out_channels": 64, "kernel_size": 5, "stride": 3},
            cnn2_cfg={"out_channels": 64, "kernel_size": 3, "stride": 1},
            fc1_cfg={"out_features": 128},
            encoded_obs_cfg={"out_features": _hidden_size},
        ),
        AGaLiTeTransformerConfig(
            in_key="encoded_obs",
            out_key="core",
            hidden_size=_hidden_size,
            n_layers=2,
            n_heads=4,
            feedforward_size=4 * _hidden_size,
            eta=4,
            r=8,
            mode="agalite",
            dropout=0.0,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_hidden_size,
            hidden_features=[1024],
            out_features=1,
            nonlinearity="ReLU",
            output_nonlinearity=None,
        ),
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        ActorQueryConfig(
            in_key="core",
            out_key="actor_query",
            hidden_size=_hidden_size,
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
