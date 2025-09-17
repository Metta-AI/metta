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
from metta.agent.lib_td.actor import (
    ActionProbs,
    ActionProbsConfig,
    ActorKey,
    ActorKeyConfig,
    ActorQuery,
    ActorQueryConfig,
)
from metta.agent.lib_td.cnn_encoder import CNNEncoder, CNNEncoderConfig
from metta.agent.lib_td.lstm import LSTM, LSTMConfig
from metta.agent.lib_td.obs_shaping import ObsShaperBox
from metta.agent.policy import Policy, PolicyArchitecture

logger = logging.getLogger(__name__)


class FastConfig(PolicyArchitecture):
    """Demonstrating that we can use config objects (ie LSTMConfig) for classes as layers (self.lstm) or attributes (ie
    actor_hidden_dim) for simple torch classes as layers (ie self.critic_1) and intermix."""

    class_path: str = "metta.agent.policies.fast.FastPolicy"

    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig()
    lstm_config: LSTMConfig = LSTMConfig()
    critic_hidden_dim: int = 1024
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig()
    actor_query_config: ActorQueryConfig = ActorQueryConfig()
    actor_key_config: ActorKeyConfig = ActorKeyConfig()
    action_probs_config: ActionProbsConfig = ActionProbsConfig()
    wants_td: bool = True


class FastPolicy(Policy):
    def __init__(self, env, config: Optional[FastConfig] = None):
        super().__init__()
        self.config = config or FastConfig()
        self.env = env
        self.wants_td = self.config.wants_td
        self.is_continuous = False
        # self.action_space = env.single_action_space
        self.action_space = env.action_space  # av used to be single action space

        self.active_action_names = []
        self.num_active_actions = 100  # Default
        self.action_index_tensor = None
        self.cum_action_max_params = None

        self.out_width = env.obs_width
        self.out_height = env.obs_height

        self.obs_shaper = ObsShaperBox(env=env)

        self.num_layers = max(env.feature_normalizations.keys()) + 1
        self.config.cnn_encoder_config.num_layers = self.num_layers
        self.cnn_encoder = CNNEncoder(config=config.cnn_encoder_config)

        self.lstm = LSTM(config=config.lstm_config)

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        module = pufferlib.pytorch.layer_init(
            nn.Linear(config.lstm_config.hidden_size, config.critic_hidden_dim), std=np.sqrt(2)
        )
        self.critic_1 = TDM(module, in_keys=["hidden"], out_keys=["critic_1"])
        module = pufferlib.pytorch.layer_init(nn.Linear(config.critic_hidden_dim, 1), std=1.0)
        self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])

        # Actor branch
        self.action_embeddings = ActionEmbedding(config=config.action_embedding_config)
        config.actor_query_config.embed_dim = config.action_embedding_config.embedding_dim
        self.actor_query = ActorQuery(config=config.actor_query_config)
        config.actor_key_config.embed_dim = config.action_embedding_config.embedding_dim
        self.actor_key = ActorKey(config=config.actor_key_config)
        self.action_probs = ActionProbs(config=config.action_probs_config)

    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        self.obs_shaper(td)
        self.cnn_encoder(td)
        self.lstm(td)
        self.critic_1(td)
        self.value_head(td)
        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td)
        self.action_probs(td, action)
        td["values"] = td["values"].flatten()

        return td

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> List[str]:
        log = self.obs_shaper.initialize_to_environment(env, device)
        self.action_embeddings.initialize_to_environment(env, device)
        self.action_probs.initialize_to_environment(env, device)
        return [log]

    def reset_memory(self):
        self.lstm.reset_memory()

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @property
    def total_params(self) -> int:
        return 0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
