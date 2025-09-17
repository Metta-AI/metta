import logging
from typing import List

from metta.agent.lib_td.action import ActionEmbeddingConfig
from metta.agent.lib_td.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.lib_td.lstm_reset import LSTMResetConfig
from metta.agent.lib_td.misc import MLPConfig
from metta.agent.lib_td.obs_enc import ObsLatentAttnConfig, ObsSelfAttnConfig
from metta.agent.lib_td.obs_shaping import ObsShaperTokensConfig
from metta.agent.lib_td.obs_tokenizers import (
    ObsAttrEmbedFourierConfig,
)
from metta.agent.policy import PolicyArchitecture
from metta.mettagrid.config import Config

logger = logging.getLogger(__name__)


class ViTSmallConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    layers: List[Config] = [
        ObsShaperTokensConfig("obs_shape"),
        ObsAttrEmbedFourierConfig(),
        ObsLatentAttnConfig(feat_dim=37, out_dim=48),
        ObsSelfAttnConfig(feat_dim=48, out_dim=128),
        LSTMResetConfig(),
        MLPConfig(name="critic", in_key="hidden", out_key="values", out_features=1, hidden_features=[1024]),
        ActionEmbeddingConfig(),
        ActorQueryConfig(),
        ActorKeyConfig(),
        ActionProbsConfig(),
    ]
