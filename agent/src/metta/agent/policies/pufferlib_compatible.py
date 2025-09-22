"""PufferLib-compatible policy that matches the exact architecture from PufferLib checkpoints."""

import logging
from typing import Optional

import numpy as np
import pufferlib.pytorch
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn

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


class PufferLibCompatibleConfig(PolicyArchitecture):
    """
    Policy configuration that exactly matches PufferLib architecture for checkpoint loading.

    Based on analysis of PufferLib checkpoint:
    - CNN: 128 channels, 24 input channels, 5x5 and 3x3 kernels
    - LSTM: 512 hidden size
    - Actor: 5 actions + 9 action args
    - Critic: Single value output
    """

    class_path: str = "metta.agent.policies.pufferlib_compatible.PufferLibCompatiblePolicy"

    # Configure for exact PufferLib match
    obs_shim_config: ObsShimBoxConfig = ObsShimBoxConfig(in_key="env_obs", out_key="obs_normalizer")
    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig(
        in_key="obs_normalizer",
        out_key="encoded_obs",
        cnn1_cfg={"out_channels": 128, "kernel_size": 5, "stride": 3},  # Match PufferLib: 128 not 64
        cnn2_cfg={"out_channels": 128, "kernel_size": 3, "stride": 1},  # Match PufferLib: 128 not 64
        fc1_cfg={"out_features": 256},  # Match PufferLib network.5: 256 features
        encoded_obs_cfg={"out_features": 256},  # Match PufferLib network.5: 256 features
    )
    lstm_config: LSTMConfig = LSTMConfig(
        in_key="encoded_obs",
        out_key="core",
        latent_size=256,  # Match PufferLib FC layer output
        hidden_size=512,  # Match PufferLib LSTM: 512 not 128
        num_layers=1,
    )

    # Match PufferLib critic and actor dimensions
    critic_hidden_dim: int = 512  # Match LSTM hidden size
    actor_hidden_dim: int = 512  # Match LSTM hidden size

    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig(out_key="action_embedding")
    actor_query_config: ActorQueryConfig = ActorQueryConfig(in_key="actor_1", out_key="actor_query")
    actor_key_config: ActorKeyConfig = ActorKeyConfig(
        query_key="actor_query", embedding_key="action_embedding", out_key="logits"
    )
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class PufferLibCompatiblePolicy(Policy):
    """Policy that exactly matches PufferLib architecture for seamless checkpoint loading."""

    def __init__(self, env, config: Optional[PufferLibCompatibleConfig] = None):
        super().__init__()
        self.config = config or PufferLibCompatibleConfig()
        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        self.active_action_names = []
        self.num_active_actions = 100  # Default
        self.action_index_tensor = None
        self.cum_action_max_params = None

        self.out_width = env.obs_width
        self.out_height = env.obs_height

        # Build components to match PufferLib exactly
        self.obs_shim = ObsShimBox(env=env, config=self.config.obs_shim_config)

        # CNN encoder with PufferLib-matching dimensions
        self.cnn_encoder = CNNEncoder(config=self.config.cnn_encoder_config, env=env)

        # LSTM with 512 hidden size to match PufferLib
        self.lstm = LSTM(config=self.config.lstm_config)

        # Actor branch - single linear layer to match PufferLib actor.0 and actor.1
        # PufferLib has: actor.0 (5 outputs) and actor.1 (9 outputs) both from 512 inputs
        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.lstm_config.hidden_size, self.config.actor_hidden_dim), std=np.sqrt(2)
        )
        self.actor_1 = TDM(module, in_keys=["core"], out_keys=["actor_1"])

        # Critic branch - single linear layer to match PufferLib value head
        # PufferLib has: value (1 output from 512 inputs)
        module = pufferlib.pytorch.layer_init(nn.Linear(self.config.lstm_config.hidden_size, 1), std=1.0)
        self.value_head = TDM(module, in_keys=["core"], out_keys=["values"])

        # Action components
        self.action_embeddings = ActionEmbedding(config=self.config.action_embedding_config)
        self.config.actor_query_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.config.actor_query_config.hidden_size = self.config.actor_hidden_dim
        self.actor_query = ActorQuery(config=self.config.actor_query_config)
        self.config.actor_key_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.actor_key = ActorKey(config=self.config.actor_key_config)
        self.action_probs = ActionProbs(config=self.config.action_probs_config)

    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        self.obs_shim(td)
        self.cnn_encoder(td)
        self.lstm(td)

        # Actor path
        self.actor_1(td)
        td["actor_1"] = torch.relu(td["actor_1"])

        # Critic path
        self.value_head(td)

        # Action processing
        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td)
        self.action_probs(td)

        return td

    @property
    def action_names(self) -> list[str]:
        """Return list of action names."""
        return getattr(self.env, "action_names", [])

    @property
    def observation_space(self):
        """Return observation space."""
        return self.env.observation_space

    def get_action_and_value(self, obs, state=None, action=None, **kwargs):
        """Get action and value prediction."""
        td = TensorDict({"env_obs": obs}, batch_size=obs.shape[:-3])
        td = self.forward(td, state=state, action=action)

        action_probs = td["action_probs"]
        values = td["values"]

        return action_probs, values, td.get("lstm_state")

    def get_value(self, obs, state=None, **kwargs):
        """Get value prediction only."""
        td = TensorDict({"env_obs": obs}, batch_size=obs.shape[:-3])
        td = self.forward(td, state=state)
        return td["values"]

    @property
    def device(self) -> torch.device:
        """Get the device this policy is on."""
        return next(self.parameters()).device

    def reset_memory(self):
        """Reset policy memory/state if any."""
        # Reset LSTM state if it has one
        if hasattr(self.lstm, 'reset_memory'):
            self.lstm.reset_memory()
