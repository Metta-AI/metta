import logging

from metta.agent.lib_td.action import ActionEmbeddingConfig
from metta.agent.lib_td.actor import ActorKeyConfig, ActorQueryConfig
from metta.agent.lib_td.lstm import LSTMConfig
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
    """Demonstrating that we can use config objects (ie LSTMConfig) for classes as layers (self.lstm) or attributes (ie
    actor_hidden_dim) for simple torch classes as layers (ie self.critic_1) and intermix."""

    obs_shaper_config: ObsShaperTokensConfig = ObsShaperTokensConfig()
    obs_attr_embed_fourier_config: ObsAttrEmbedFourierConfig = ObsAttrEmbedFourierConfig()
    obs_latent_attn_config: ObsLatentAttnConfig = ObsLatentAttnConfig()
    obs_self_attn_config: ObsSelfAttnConfig = ObsSelfAttnConfig()
    obs_self_attn_config.feat_dim = 48
    lstm_config: LSTMConfig = LSTMConfig()
    critic_config: MLPConfig = MLPConfig("critic", "hidden", "values", 1, [1024])
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig()
    actor_query_config: ActorQueryConfig = ActorQueryConfig()
    actor_key_config: ActorKeyConfig = ActorKeyConfig()

    def instantiate(self, env, obs_meta: dict):
        return Policy(env, obs_meta, config=self)


# class ViTLSTMPolicy(nn.Module):
#     def __init__(self, env, obs_meta: dict, config: Optional[ViTLSTMConfig] = None):
#         super().__init__()
#         self.config = config or ViTLSTMConfig()
#         self.obs_meta = obs_meta
#         self.wants_td = True
#         self.is_continuous = False
#         self.action_space = env.single_action_space

#         self.active_action_names = []
#         self.num_active_actions = 100  # Default
#         self.action_index_tensor = None
#         self.cum_action_max_params = None

#         self.out_width = obs_meta["obs_width"]
#         self.out_height = obs_meta["obs_height"]

#         # Dynamically determine num_layers from environment features
#         # This matches what ComponentPolicy does via ObsTokenToBoxShaper
#         self.num_layers = max(obs_meta["feature_normalizations"].keys()) + 1

#         # Match YAML component initialization more closely
#         # Use dynamically determined num_layers as input channels
#         # Note: YAML uses orthogonal with gain=1, not sqrt(2) like pufferlib default
#         self.obs_token_pad_strip = ObsTokenPadStrip(config=config.obs_token_pad_strip)

#         self.lstm = LSTM(config=config.lstm_config)

#         # Critic branch
#         # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
#         module = pufferlib.pytorch.layer_init(
#             nn.Linear(config.lstm_config.hidden_size, config.critic_hidden_dim), std=np.sqrt(2)
#         )
#         self.critic_1 = TDM(module, in_keys=["hidden"], out_keys=["critic_1"])
#         # value_head has no nonlinearity (YAML: nonlinearity: null), so gain=1
#         module = pufferlib.pytorch.layer_init(nn.Linear(config.critic_hidden_dim, 1), std=1.0)
#         self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])

#         # Actor branch
#         # actor_1 uses gain=1 (YAML default for Linear layers with ReLU)
#         module = pufferlib.pytorch.layer_init(
#             nn.Linear(config.lstm_config.hidden_size, config.actor_hidden_dim), std=1.0
#         )
#         self.actor_1 = TDM(module, in_keys=["hidden"], out_keys=["actor_1"])

#         # Action embeddings - will be properly initialized via activate_action_embeddings
#         self.action_embeddings = ActionEmbedding(config=config.action_embedding_config)

#         # Bilinear layer to match MettaActorSingleHead
#         config.actor_query_config.embed_dim = config.action_embedding_config.embedding_dim
#         self.actor_query = ActorQuery(config=config.actor_query_config)
#         config.actor_key_config.embed_dim = config.action_embedding_config.embedding_dim
#         self.actor_key = ActorKey(config=config.actor_key_config)

#     def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
#         self.cnn_encoder(td)
#         self.lstm(td)
#         self.critic_1(td)
#         self.value_head(td)
#         self.actor_1(td)
#         self.action_embeddings(td)
#         self.actor_query(td)
#         self.actor_key(td, action)
#         td["values"] = td["values"].flatten()

#         return td

#     # you need to expose the update methods of your policy to metta agent. We could instead have a self.components
#     # dict and then run hasattr, letting metta agent walk the tree.
#     def initialize_to_environment(self, full_action_names: list[str], device):
#         """Initialize to environment, setting up action embeddings to match the available actions."""
#         self.active_action_names = full_action_names
#         self.num_active_actions = len(full_action_names)
#         self.action_embeddings.initialize_to_environment(full_action_names, device)
#         assert self.action_index_tensor is not None and self.cum_action_max_params is not None
#         self.actor_key.initialize_to_environment(self.action_index_tensor, self.cum_action_max_params)

#     def update_normalization_factors(self, features: dict[str, dict]):
#         """You need a sequence encoder policy for this."""
#         pass

#     def _update_normalization_factors(self, features: dict[str, dict]):
#         self.cnn_encoder.obs_normalizer.update_normalization_factors(features)

#     def reset_memory(self):
#         self.lstm.reset_memory()
