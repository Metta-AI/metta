import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import pufferlib.pytorch
from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action as MettaGridAction
from mettagrid.simulator import AgentObservation as MettaGridObservation

logger = logging.getLogger("mettagrid.policy.token_policy")


def coordinates(observations: torch.Tensor, dtype: torch.dtype = torch.uint8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split packed observation bytes into (x, y) nibble indices as ``dtype`` tensors."""

    coords_byte = observations[..., 0].to(torch.long)
    y_coords = (coords_byte >> 4) & 0x0F
    x_coords = coords_byte & 0x0F
    return x_coords.to(dtype), y_coords.to(dtype)


class TokenPolicyNet(torch.nn.Module):
    """Token-aware per-step encoder inspired by Metta's basic baseline."""

    def __init__(self, features: list[ObservationFeatureSpec], actions_cfg: ActionsConfig):
        super().__init__()

        self.hidden_size = 192

        feature_norms = {feature.id: feature.normalization for feature in features}
        max_feature_id = max((int(feature_id) for feature_id in feature_norms.keys()), default=-1)
        num_feature_embeddings = max(256, max_feature_id + 1)
        feature_scale = torch.ones(num_feature_embeddings, dtype=torch.float32)
        for feature_id, norm in feature_norms.items():
            feature_scale[feature_id] = max(float(norm), 1.0)

        self.register_buffer("_feature_scale", feature_scale)

        self.pos_x_embed = nn.Embedding(256, self.hidden_size)
        self.pos_y_embed = nn.Embedding(256, self.hidden_size)
        self.feature_embed = nn.Embedding(num_feature_embeddings, self.hidden_size)

        self.token_mlp = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.post_mlp = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.ReLU(),
            nn.Dropout(0.1),
            pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.ReLU(),
        )

        self.num_actions = len(actions_cfg.actions())

        self.action_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.num_actions))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1))

    def _flatten_tokens(self, observations: torch.Tensor) -> torch.Tensor:
        """Ensure observations are shaped as (batch, tokens, 3)."""

        if observations.dim() == 2:
            observations = observations.unsqueeze(0)

        if observations.dim() == 4:
            observations = observations.flatten(0, 1)
        elif observations.dim() != 3:
            raise ValueError(f"Unexpected observation shape {tuple(observations.shape)}; expected (*, tokens, 3).")

        return observations

    def _encode_tokens(self, observations: torch.Tensor) -> torch.Tensor:
        tokens = self._flatten_tokens(observations)

        coords_byte = tokens[..., 0].to(torch.long)
        x_coords, y_coords = coordinates(tokens, torch.long)
        feature_ids = tokens[..., 1].to(torch.long)
        values = tokens[..., 2].to(torch.float32)

        valid_mask = coords_byte != 0xFF

        x_coords = torch.clamp(x_coords, min=0, max=255)
        y_coords = torch.clamp(y_coords, min=0, max=255)
        feature_ids_clamped = torch.clamp(feature_ids, min=0, max=self.feature_embed.num_embeddings - 1)

        token_embed = self.pos_x_embed(x_coords) + self.pos_y_embed(y_coords) + self.feature_embed(feature_ids_clamped)

        scale = self._feature_scale[feature_ids_clamped]
        scaled_values = (values / (scale + 1e-6)).unsqueeze(-1)

        token_embed = token_embed * scaled_values
        token_embed = token_embed * valid_mask.unsqueeze(-1).to(token_embed.dtype)

        summary = token_embed.sum(dim=-2)
        token_counts = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(torch.float32)
        summary = summary / torch.sqrt(token_counts)

        return self.token_mlp(summary)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._encode_tokens(observations)
        hidden = self.post_mlp(features)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_eval(observations, state)


class TokenAgentPolicyImpl(AgentPolicy):
    """Per-agent policy utilising the shared token encoder network."""

    def __init__(self, net: TokenPolicyNet, device: torch.device, num_actions: int):
        self._net = net
        self._device = device
        self._num_actions = num_actions

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        obs_tensor = torch.tensor(obs, device=self._device).unsqueeze(0).float()

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            return dtype_actions.type(action)


class TokenPolicy(MultiAgentPolicy):
    """Feed-forward token encoder baseline derived from Metta's token-based basic policy."""

    short_names = ["token"]

    def __init__(
        self,
        features: list[ObservationFeatureSpec],
        actions_cfg: ActionsConfig,
        device: str,
        policy_env_info: PolicyEnvInterface,
    ):
        super().__init__(policy_env_info, device=device)

        torch_device = torch.device(device)
        self._net = TokenPolicyNet(features, actions_cfg).to(torch_device)
        self._device = torch_device
        self._num_actions = len(actions_cfg.actions())

    def network(self) -> nn.Module:
        """Return the underlying token encoder network for training."""
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return TokenAgentPolicyImpl(self._net, self._device, self._num_actions)

    def is_recurrent(self) -> bool:
        return False

    def load_policy_data(self, checkpoint_path: str) -> None:
        """Load token encoder network weights from file."""
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        self._net.load_state_dict(state_dict)
        self._net = self._net.to(self._device)

    def save_policy_data(self, checkpoint_path: str) -> None:
        """Save token encoder network weights to file."""
        torch.save(self._net.state_dict(), checkpoint_path)
