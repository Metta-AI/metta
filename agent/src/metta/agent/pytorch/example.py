import logging
from typing import Optional

import einops
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.pytorch.base import LSTMWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class Example(PyTorchAgentMixin, LSTMWrapper):
    """Recurrent LSTM-based policy wrapper with discrete multi-head action space."""

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        cnn_channels: int = 128,
        input_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 2,
        **kwargs,
    ):
        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)

        if policy is None:
            policy = Policy(env, cnn_channels=cnn_channels, hidden_size=hidden_size, input_size=input_size)

        # Use enhanced LSTMWrapper with num_layers support
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)

    def forward(self, td: TensorDict, state: Optional[dict] = None, action=None) -> TensorDict:
        """Forward pass: encodes observations, runs LSTM, decodes into actions, value, and stats."""

        observations = td["env_obs"]

        # Use mixin to set critical TensorDict fields
        B, TT = self.set_tensordict_fields(td, observations)

        # Handle BPTT reshaping if needed
        if td.batch_dims > 1:
            total_batch = B * TT
            td = td.reshape(total_batch)
        state = state or {"lstm_h": None, "lstm_c": None, "hidden": None}

        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        lstm_state = self._prepare_lstm_state(state)

        hidden = hidden.view(B, TT, -1).transpose(0, 1)
        lstm_output, (new_h, new_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        logits_list, value = self.policy.decode_actions(flat_hidden)
        actions, log_probs, entropies, full_log_probs = self._sample_actions(logits_list)

        if len(actions) >= 2:
            actions_tensor = torch.stack([actions[0], actions[1]], dim=-1)
        else:
            actions_tensor = torch.stack([actions[0], torch.zeros_like(actions[0])], dim=-1)
        actions_tensor = actions_tensor.to(dtype=torch.int32)

        if action is None:
            td["actions"] = torch.zeros(actions_tensor.shape, dtype=torch.int32, device=observations.device)
            td["act_log_prob"] = log_probs.mean(dim=-1)
            td["values"] = value.flatten()
            td["full_log_probs"] = full_log_probs
        else:
            td["act_log_prob"] = log_probs.mean(dim=-1)
            td["entropy"] = entropies.sum(dim=-1)
            td["value"] = value.flatten()
            td["full_log_probs"] = full_log_probs
            td = td.reshape(B, TT)
        return td

    def _sample_actions(self, logits_list: list[torch.Tensor]):
        """Samples discrete actions from logits and computes log-probs and entropy."""
        actions, selected_log_probs, entropies, full_log_probs = [], [], [], []
        max_actions = max(logits.shape[1] for logits in logits_list)

        for logits in logits_list:
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            action = torch.multinomial(probs, 1).squeeze(-1)
            batch_idx = torch.arange(action.shape[0], device=action.device)

            selected_log_prob = log_probs[batch_idx, action]
            entropy = -(probs * log_probs).sum(dim=-1)

            actions.append(action)
            selected_log_probs.append(selected_log_prob)
            entropies.append(entropy)
            pad_width = max_actions - log_probs.shape[1]
            full_log_probs.append(F.pad(log_probs, (0, pad_width), value=float("-inf")))

        return (
            actions,
            torch.stack(selected_log_probs, dim=-1),
            torch.stack(entropies, dim=-1),
            torch.stack(full_log_probs, dim=-1),
        )


class Policy(nn.Module):
    """CNN + Self feature encoder policy for discrete multi-head action space."""

    def __init__(self, env, cnn_channels=128, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_space = env.single_action_space
        self.is_continuous = kwargs.get("is_continuous", False)

        self.out_width, self.out_height, self.num_layers = 11, 11, 22

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.num_layers, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, hidden_size // 2)),
            nn.ReLU(),
        )

        # Use values that avoid division by very small numbers
        # These values represent the expected maximum for each feature layer
        max_vec = torch.tensor(
            [
                9.0,
                1.0,
                1.0,
                10.0,
                3.0,
                254.0,
                1.0,
                1.0,
                235.0,
                8.0,
                9.0,
                250.0,
                29.0,
                1.0,
                1.0,
                8.0,
                1.0,
                1.0,
                6.0,
                3.0,
                1.0,
                2.0,
            ],
            dtype=torch.float32,
        )
        # Clamp minimum value to 1.0 to avoid near-zero divisions
        max_vec = torch.maximum(max_vec, torch.ones_like(max_vec))
        max_vec = max_vec[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        self.actor = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in self.action_space.nvec]
        )
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
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
        atr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
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

        batch_idx = torch.arange(B * TT, device=observations.device).unsqueeze(-1).expand_as(atr_values)
        box_obs[batch_idx[valid_tokens], atr_indices[valid_tokens], x_coords[valid_tokens], y_coords[valid_tokens]] = (
            atr_values[valid_tokens]
        )

        # Normalize features with epsilon for numerical stability
        features = box_obs / (self.max_vec + 1e-8)
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)

        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden: torch.Tensor):
        """Maps hidden features to action logits and value."""
        return [head(hidden) for head in self.actor], self.value(hidden)
