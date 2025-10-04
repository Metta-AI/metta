import logging
from typing import List, Optional

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn
from torchrl.data import Composite, UnboundedDiscrete

import pufferlib.pytorch
from metta.agent.components.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.components.actor import (
    ActionProbs,
    ActionProbsConfig,
    ActorKey,
    ActorKeyConfig,
    ActorQuery,
    ActorQueryConfig,
)
from metta.agent.components.cnn_encoder import CNNEncoder, CNNEncoderConfig
from metta.agent.components.lstm import LSTM, LSTMConfig
from metta.agent.components.obs_shim import ObsShimBox, ObsShimBoxConfig
from metta.agent.policy import Policy, PolicyArchitecture

logger = logging.getLogger(__name__)


class FastConfig(PolicyArchitecture):
    """
    Fast uses a CNN encoder so is not flexible to changing observation features but it runs faster than the ViT encoder.

    This particular class is also set up without using PolicyAutoBuilder to demonstrate an alternative way to build a
    policy, affording more control over the process at the expense of more code. It demonstrates that we can use config
    objects (ie LSTMConfig) for classes as layers (self.lstm) or attributes (ie actor_hidden_dim) for simple torch
    classes as layers (ie self.critic_1), wrap them in a TensorDictModule, and intermix."""

    class_path: str = "metta.agent.policies.fast.FastPolicy"

    obs_shim_config: ObsShimBoxConfig = ObsShimBoxConfig(in_key="env_obs", out_key="obs_normalizer")
    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig(in_key="obs_normalizer", out_key="encoded_obs")
    lstm_config: LSTMConfig = LSTMConfig(
        in_key="encoded_obs", out_key="core", latent_size=128, hidden_size=128, num_layers=2
    )
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig(out_key="action_embedding")
    actor_query_config: ActorQueryConfig = ActorQueryConfig(in_key="actor_1", out_key="actor_query")
    actor_key_config: ActorKeyConfig = ActorKeyConfig(
        query_key="actor_query", embedding_key="action_embedding", out_key="logits"
    )
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class FastPolicy(Policy):
    def __init__(self, env, config: Optional[FastConfig] = None):
        super().__init__()
        self.config = config or FastConfig()
        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        self.active_action_names = []
        self.num_active_actions = 100  # Default

        self.out_width = env.obs_width
        self.out_height = env.obs_height

        self.obs_shim = ObsShimBox(env=env, config=self.config.obs_shim_config)

        self.cnn_encoder = CNNEncoder(config=self.config.cnn_encoder_config, env=env)

        self.lstm = LSTM(config=self.config.lstm_config)

        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.lstm_config.hidden_size, self.config.actor_hidden_dim), std=1.0
        )
        self.actor_1 = TDM(module, in_keys=["core"], out_keys=["actor_1"])

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.lstm_config.hidden_size, self.config.critic_hidden_dim), std=np.sqrt(2)
        )
        self.critic_1 = TDM(module, in_keys=["core"], out_keys=["critic_1"])
        self.critic_activation = nn.Tanh()
        module = pufferlib.pytorch.layer_init(nn.Linear(self.config.critic_hidden_dim, 1), std=1.0)
        self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])

        # Actor branch
        self.action_embeddings = ActionEmbedding(config=self.config.action_embedding_config)
        self.config.actor_query_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.config.actor_query_config.hidden_size = self.config.actor_hidden_dim
        self.actor_query = ActorQuery(config=self.config.actor_query_config)
        self.config.actor_key_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.actor_key = ActorKey(config=self.config.actor_key_config)
        self.action_probs = ActionProbs(config=self.config.action_probs_config)

    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations hurting performance
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        self.obs_shim(td)
        self.cnn_encoder(td)
        self.lstm(td)
        self.actor_1(td)
        td["actor_1"] = torch.relu(td["actor_1"])
        self.critic_1(td)
        td["critic_1"] = self.critic_activation(td["critic_1"])
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
        device = torch.device(device)
        self.to(device)

        log = self.obs_shim.initialize_to_environment(env, device)
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
    def device(self) -> torch.device:
        return next(self.parameters()).device
