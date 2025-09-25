import logging
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn

import pufferlib.pytorch
from metta.agent.lib_td.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.lib_td.actor import ActorKey, ActorKeyConfig, ActorQuery, ActorQueryConfig
from metta.agent.lib_td.cnn_encoder import CNNEncoder, CNNEncoderConfig
from metta.agent.lib_td.transformer import Transformer, TransformerConfig
from metta.common.config.config import Config

logger = logging.getLogger(__name__)


class FastTransformerConfig(Config):
    """Fast policy variant that uses a causal Transformer instead of an LSTM."""

    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig()
    transformer_config: TransformerConfig = TransformerConfig()
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig()
    actor_query_config: ActorQueryConfig = ActorQueryConfig()
    actor_key_config: ActorKeyConfig = ActorKeyConfig()
    wants_td: bool = True

    def instantiate(self, env, obs_meta: dict):
        return FastTransformerPolicy(env, obs_meta, config=self)


class FastTransformerPolicy(nn.Module):
    def __init__(self, env, obs_meta: dict, config: Optional[FastTransformerConfig] = None):
        super().__init__()
        self.config = config or FastTransformerConfig()
        self.obs_meta = obs_meta
        self.wants_td = self.config.wants_td
        self.is_continuous = False
        self.action_space = env.single_action_space

        self.active_action_names = []
        self.num_active_actions = 100  # Default
        self.action_index_tensor = None
        self.cum_action_max_params = None

        self.out_width = obs_meta["obs_width"]
        self.out_height = obs_meta["obs_height"]

        # Dynamically determine num_layers from environment features
        self.num_layers = max(obs_meta["feature_normalizations"].keys()) + 1

        # Encoder produces `encoded_obs` sized to feed the sequence model
        self.cnn_encoder = CNNEncoder(obs_meta, config=self.config.cnn_encoder_config)

        # Sequence model: Transformer with API compatible outputs via out_key="hidden"
        self.transformer = Transformer(config=self.config.transformer_config)

        # Critic branch
        model_hidden = self.config.transformer_config.hidden_size
        module = pufferlib.pytorch.layer_init(nn.Linear(model_hidden, self.config.critic_hidden_dim), std=np.sqrt(2))
        self.critic_1 = TDM(module, in_keys=["hidden"], out_keys=["critic_1"])
        module = pufferlib.pytorch.layer_init(nn.Linear(self.config.critic_hidden_dim, 1), std=1.0)
        self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])

        # Actor branch
        module = pufferlib.pytorch.layer_init(nn.Linear(model_hidden, self.config.actor_hidden_dim), std=1.0)
        self.actor_1 = TDM(module, in_keys=["hidden"], out_keys=["actor_1"])

        # Action embeddings and attention scoring
        self.action_embeddings = ActionEmbedding(config=self.config.action_embedding_config)

        # Ensure bilinear dims match embeddings
        self.config.actor_query_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.actor_query = ActorQuery(config=self.config.actor_query_config)
        self.config.actor_key_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.actor_key = ActorKey(config=self.config.actor_key_config)

    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        self.cnn_encoder(td)
        self.transformer(td)
        self.critic_1(td)
        self.value_head(td)
        self.actor_1(td)
        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td, action)
        td["values"] = td["values"].flatten()

        return td

    def initialize_to_environment(self, full_action_names: list[str], device):
        """Initialize to environment, setting up action embeddings to match the available actions."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)
        self.action_embeddings.initialize_to_environment(full_action_names, device)
        assert self.action_index_tensor is not None and self.cum_action_max_params is not None
        self.actor_key.initialize_to_environment(self.action_index_tensor, self.cum_action_max_params)

    def update_normalization_factors(self, features: dict[str, dict]):
        """You need a sequence encoder policy for this."""
        pass

    def _update_normalization_factors(self, features: dict[str, dict]):
        self.cnn_encoder.obs_normalizer.update_normalization_factors(features)

    def reset_memory(self):
        self.transformer.reset_memory()
