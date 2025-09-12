import logging
from typing import List, Optional

import numpy as np
import pufferlib.pytorch
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.lib_td.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.lib_td.actor import ActorKey, ActorKeyConfig, ActorQuery, ActorQueryConfig
from metta.agent.lib_td.cnn_encoder import CNNEncoder, CNNEncoderConfig
from metta.agent.lib_td.lstm import LSTM, LSTMConfig
from metta.agent.lib_td.obs_shaping import ObsShaperBoxConfig
from metta.common.config.config import Config
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


class FastConfig(Config):
    """Demonstrating that we can use config objects (ie LSTMConfig) for classes as layers (self.lstm) or attributes (ie
    actor_hidden_dim) for simple torch classes as layers (ie self.critic_1) and intermix."""

    obs_shaper_config: ObsShaperBoxConfig = ObsShaperBoxConfig()
    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig()
    lstm_config: LSTMConfig = LSTMConfig()
    critic_hidden_dim: int = 1024
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig()
    actor_query_config: ActorQueryConfig = ActorQueryConfig()
    actor_key_config: ActorKeyConfig = ActorKeyConfig()
    wants_td: bool = True

    def instantiate(self, env, obs_meta: dict):
        return FastPolicy(env, obs_meta, config=self)


class FastPolicy(nn.Module):
    def __init__(self, env, obs_meta: dict, config: Optional[FastConfig] = None):
        super().__init__()
        self.config = config or FastConfig()
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
        # This matches what ComponentPolicy does via ObsTokenToBoxShaper
        self.num_layers = max(obs_meta["feature_normalizations"].keys()) + 1

        # Match YAML component initialization more closely
        # Use dynamically determined num_layers as input channels
        # Note: YAML uses orthogonal with gain=1, not sqrt(2) like pufferlib default
        self.cnn_encoder = CNNEncoder(obs_meta, config=config.cnn_encoder_config)

        self.lstm = LSTM(config=config.lstm_config)

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        module = pufferlib.pytorch.layer_init(
            nn.Linear(config.lstm_config.hidden_size, config.critic_hidden_dim), std=np.sqrt(2)
        )
        self.critic_1 = TDM(module, in_keys=["hidden"], out_keys=["critic_1"])
        # value_head has no nonlinearity (YAML: nonlinearity: null), so gain=1
        module = pufferlib.pytorch.layer_init(nn.Linear(config.critic_hidden_dim, 1), std=1.0)
        self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])

        # Actor branch
        # Action embeddings - will be properly initialized via activate_action_embeddings
        self.action_embeddings = ActionEmbedding(config=config.action_embedding_config)

        # Bilinear layer to match MettaActorSingleHead
        config.actor_query_config.embed_dim = config.action_embedding_config.embedding_dim
        self.actor_query = ActorQuery(config=config.actor_query_config)
        config.actor_key_config.embed_dim = config.action_embedding_config.embedding_dim
        self.actor_key = ActorKey(config=config.actor_key_config)

    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        self.cnn_encoder(td)
        self.lstm(td)
        self.critic_1(td)
        self.value_head(td)
        self.actor_1(td)
        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td, action)
        td["values"] = td["values"].flatten()

        return td

    # you need to expose the update methods of your policy to metta agent. We could instead have a self.components
    # dict and then run hasattr, letting metta agent walk the tree.
    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = None,
    ) -> List[str]:
        log = self.obs_shaper.initialize_to_environment(features, action_names, action_max_params, device, is_training)
        self.action_embeddings.initialize_to_environment(features, action_names, action_max_params, device, is_training)
        self.actor_key.initialize_to_environment(features, action_names, action_max_params, device, is_training)
        return [log]

    def _apply_feature_remapping(self, features: dict[str, dict]):
        """You need a sequence encoder policy for this."""
        pass

    def reset_memory(self):
        self.lstm.reset_memory()

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            actions=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            last_actions=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def attach_replay_buffer(self, experience: Experience):
        """Losses expect to find a replay buffer in the policy."""
        self.replay = experience


# av need Distributed Metta Agent class to wrap this
