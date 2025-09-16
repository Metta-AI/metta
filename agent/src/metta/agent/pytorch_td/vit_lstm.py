import logging

from metta.agent.lib_td.action import ActionEmbeddingConfig
from metta.agent.lib_td.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.lib_td.lstm_reset import LSTMResetConfig
from metta.agent.lib_td.misc import MLPConfig
from metta.agent.lib_td.obs_enc import ObsLatentAttnConfig, ObsSelfAttnConfig
from metta.agent.lib_td.obs_shaping import ObsShaperTokensConfig
from metta.agent.lib_td.obs_tokenizers import (
    ObsAttrEmbedFourierConfig,
)
from metta.agent.pytorch_td.policy import Policy
from metta.common.config.config import Config

logger = logging.getLogger(__name__)


class ViTLSTMConfig(Config):
    obs_shaper_config: ObsShaperTokensConfig = ObsShaperTokensConfig()
    obs_attr_embed_fourier_config: ObsAttrEmbedFourierConfig = ObsAttrEmbedFourierConfig()
    obs_latent_attn_config: ObsLatentAttnConfig = ObsLatentAttnConfig(feat_dim=37, out_dim=48)
    obs_self_attn_config: ObsSelfAttnConfig = ObsSelfAttnConfig(feat_dim=48, out_dim=128)
    lstm_config: LSTMResetConfig = LSTMResetConfig()
    critic_config: MLPConfig = MLPConfig(
        name="critic", in_key="hidden", out_key="values", out_features=1, hidden_features=[1024]
    )
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig()
    actor_query_config: ActorQueryConfig = ActorQueryConfig()
    actor_key_config: ActorKeyConfig = ActorKeyConfig()
    action_probs_config: ActionProbsConfig = ActionProbsConfig()

    def instantiate(self, env):
        return Policy(env, config=self)
