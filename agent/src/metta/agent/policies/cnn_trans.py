import logging
from typing import List

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.cnn_encoder import CNNEncoderConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.lstm import LSTMConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_shim import ObsShimBoxConfig
from metta.agent.policy import PolicyArchitecture

logger = logging.getLogger(__name__)


class CNNTransConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 128
    _embedding_dim = 16

    components: List[ComponentConfig] = [
        ObsShimBoxConfig(in_key="env_obs", out_key="obs_shim_box"),
        CNNEncoderConfig(in_key="obs_shim_box", out_key="obs_cnn_encoder"),
        # VanillaTransformerConfig(in_key="obs_cnn_encoder", out_key="core", embed_dim=_embed_dim),
        LSTMConfig(in_key="obs_cnn_encoder", out_key="core", embed_dim=_embed_dim),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_embed_dim,
            out_features=1,
            hidden_features=[1024],
        ),
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embedding_dim),
        ActorQueryConfig(in_key="core", out_key="actor_query", hidden_size=_embed_dim, embed_dim=_embedding_dim),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embedding_dim,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")
