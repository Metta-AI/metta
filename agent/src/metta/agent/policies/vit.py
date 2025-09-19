import logging
from typing import List

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.lstm_reset import LSTMResetConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsLatentAttnConfig, ObsSelfAttnConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import (
    ObsAttrEmbedFourierConfig,
)
from metta.agent.policy import PolicyArchitecture

logger = logging.getLogger(__name__)


class ViTSmallConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens"),
        ObsAttrEmbedFourierConfig(in_key="obs_shim_tokens", out_key="obs_attr_embed_fourier"),
        ObsLatentAttnConfig(in_key="obs_attr_embed_fourier", out_key="obs_latent_attn", feat_dim=37, out_dim=48),
        ObsSelfAttnConfig(in_key="obs_latent_attn", out_key="obs_self_attn", feat_dim=48, out_dim=128),
        LSTMResetConfig(in_key="obs_self_attn", out_key="core", latent_size=128, hidden_size=128, num_layers=2),
        MLPConfig(in_key="core", out_key="values", name="critic", out_features=1, hidden_features=[1024]),
        ActionEmbeddingConfig(out_key="action_embedding"),
        ActorQueryConfig(in_key="core", out_key="actor_query"),
        ActorKeyConfig(query_key="actor_query", embedding_key="action_embedding", out_key="logits"),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")
