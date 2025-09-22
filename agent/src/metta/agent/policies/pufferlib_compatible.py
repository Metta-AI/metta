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
from metta.agent.components.cnn_encoder import CNNEncoderConfig
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


        self.conv1 = nn.Conv2d(24, 128, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1)

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
        )

        self.max_vec = [1.0] * 24
        for feature_id, norm_value in self.env.feature_normalizations.items():
            if feature_id < 24:
                self.max_vec[feature_id] = norm_value if norm_value > 0 else 1.0
        self.max_vec = torch.tensor(self.max_vec, dtype=torch.float32)
        self.max_vec = torch.maximum(self.max_vec, torch.ones_like(self.max_vec))
        self.max_vec = self.max_vec[None, :, None, None]
        self.register_buffer("max_vec", self.max_vec)

        action_nvec = self.env.single_action_space.nvec
        self.actor = nn.ModuleList(
            [
                nn.Linear(256, n)
                for n in action_nvec
            ]
        )
        self.value = nn.Linear(256, 1)



        # # LSTM with 512 hidden size to match PufferLib
        # self.lstm = LSTM(config=self.config.lstm_config)

    def encode_observations(
        self, observations: torch.Tensor, state=None
    ) -> torch.Tensor:
        """Converts raw observation tokens into a concatenated self + CNN feature vector."""
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")

        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range, but we need to make them long for indexing)
        x_coords = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coords = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = observations[
            ..., 1
        ].long()  # Shape: [B_TT, M], ready for embedding
        atr_values = observations[..., 2].float()  # Shape: [B_TT, M]

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=observations.device,
        )

        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )


        batch_idx = (
            torch.arange(B * TT, device=observations.device)
            .unsqueeze(-1)
            .expand_as(atr_values)
        )
        box_obs[
            batch_idx[valid_tokens],
            atr_indices[valid_tokens],
            x_coords[valid_tokens],
            y_coords[valid_tokens],
        ] = atr_values[valid_tokens]

        # Normalize features with epsilon for numerical stability
        features = box_obs / (self.max_vec + 1e-8)

        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        result = torch.cat([self_features, cnn_features], dim=1)
        return result
    

    def decode_actions(self, hidden):
        #hidden = self.layer_norm(hidden)
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value



    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        
        observations = td["env_obs"]
        hidden = self.encode_observations(observations)
        logits, value = self.decode_actions(hidden)
        td["action_probs"] = logits
        td["values"] = value

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
        if hasattr(self.lstm, "reset_memory"):
            self.lstm.reset_memory()
