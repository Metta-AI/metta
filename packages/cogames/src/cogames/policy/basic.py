import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation

logger = logging.getLogger("cogames.policies.basic_policy")


class BasicPolicyNet(torch.nn.Module):
    """Token-aware per-step encoder inspired by Metta's basic baseline."""

    def __init__(self, env: MettaGridEnv):
        super().__init__()

        self.hidden_size = 192

        feature_norms = getattr(env, "feature_normalizations", {})
        max_feature_id = max((int(feature_id) for feature_id in feature_norms.keys()), default=-1)
        num_feature_embeddings = max(256, max_feature_id + 1)
        feature_scale = torch.ones(num_feature_embeddings, dtype=torch.float32)
        for feature_id, norm in feature_norms.items():
            index = int(feature_id)
            if 0 <= index < num_feature_embeddings:
                feature_scale[index] = max(float(norm), 1.0)

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

        self.action_nvec = tuple(env.single_action_space.nvec)

        self.action_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, sum(self.action_nvec)))
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
        feature_ids = tokens[..., 1].to(torch.long)
        values = tokens[..., 2].to(torch.float32)

        valid_mask = coords_byte != 0xFF

        x_coords = torch.clamp((coords_byte >> 4) & 0x0F, min=0, max=255)
        y_coords = torch.clamp(coords_byte & 0x0F, min=0, max=255)
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
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = self._encode_tokens(observations)
        hidden = self.post_mlp(features)
        logits = self.action_head(hidden)
        logits_split = logits.split(self.action_nvec, dim=1)
        values = self.value_head(hidden)
        return list(logits_split), values

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        return self.forward_eval(observations, state)


class BasicAgentPolicyImpl(AgentPolicy):
    """Per-agent policy utilising the shared token encoder network."""

    def __init__(self, net: BasicPolicyNet, device: torch.device, action_nvec: tuple[int, ...]):
        self._net = net
        self._device = device
        self._action_nvec = action_nvec

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        obs_tensor = torch.tensor(obs, device=self._device).unsqueeze(0).float()

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)

            actions: list[int] = []
            for logit in logits:
                dist = torch.distributions.Categorical(logits=logit)
                actions.append(dist.sample().item())

            return np.array(actions, dtype=np.int32)


class BasicPolicy(TrainablePolicy):
    """Feed-forward token encoder baseline derived from Metta's basic policy."""

    def __init__(self, env: MettaGridEnv, device: torch.device):
        super().__init__()
        self._net = BasicPolicyNet(env).to(device)
        self._device = device
        self._action_nvec = tuple(env.single_action_space.nvec)

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return BasicAgentPolicyImpl(self._net, self._device, self._action_nvec)

    def load_policy_data(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        self._net.load_state_dict(state_dict)
        self._net = self._net.to(self._device)

    def save_policy_data(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
