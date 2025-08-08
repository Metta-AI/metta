import logging
from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import nn

import pufferlib.models
import pufferlib.pytorch

logger = logging.getLogger(__name__)


class Recurrent(pufferlib.models.LSTMWrapper):
    """Recurrent LSTM-based policy wrapper with discrete multi-head action space."""

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        cnn_channels: int = 128,
        input_size: int = 512,
        hidden_size: int = 512,
    ):
        if policy is None:
            policy = Policy(
                env,
                cnn_channels=cnn_channels,
                hidden_size=hidden_size,
                input_size=input_size,
            )
        super().__init__(env, policy, input_size, hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ) -> None:
        """Sets up action space mappings for the environment."""
        self.activate_actions(action_names, action_max_params, device)

    def activate_actions(
        self,
        action_names: list[str],
        action_max_params: list[int],
        device: torch.device,
    ) -> None:
        """Initialize discrete action heads and precompute indexing tables."""
        assert isinstance(action_max_params, list), "action_max_params must be a list"
        self.device = device
        self.action_names = action_names
        self.action_max_params = action_max_params
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Cumulative indices for mapping actions
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=device, dtype=torch.long), dim=0
        )

        # Build action index tensor [action_type, param]
        action_index = [
            [atype, param]
            for atype, max_param in enumerate(action_max_params)
            for param in range(max_param + 1)
        ]
        self.action_index_tensor = torch.tensor(action_index, device=device, dtype=torch.int32)

        logger.info(f"Initialized policy actions: {self.active_actions}")

    def forward(self, observations: torch.Tensor, state: Optional[dict] = None, action=None) -> dict:
        """Forward pass: encodes observations, runs LSTM, decodes into actions, value, and stats."""
        state = state or {"lstm_h": None, "lstm_c": None, "hidden": None}
        observations = observations.to(self.device)

        # Encode observations
        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # Prepare LSTM state
        lstm_state = self._prepare_lstm_state(state)

        # Run LSTM
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, input_size)
        lstm_output, (new_h, new_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode actions and values
        logits_list, value = self.policy.decode_actions(flat_hidden)

        actions, log_probs, entropies, full_log_probs = self._sample_actions(logits_list)

        # Ensure at least 2D action tensor
        if len(actions) >= 2:
            actions_tensor = torch.stack([actions[0], actions[1]], dim=-1)
        else:
            actions_tensor = torch.stack([actions[0], torch.zeros_like(actions[0])], dim=-1)

        return {
            "actions": actions_tensor,
            "act_log_prob": log_probs.mean(dim=-1),
            "entropy": entropies.sum(dim=-1),
            "value": value.flatten(),
            "full_log_probs": full_log_probs,
            "env_obs": observations,
        }

    def _prepare_lstm_state(self, state: dict):
        """Ensures LSTM hidden states are on the correct device and sized properly."""
        h, c = state.get("lstm_h"), state.get("lstm_c")
        if h is None or c is None:
            return None
        h, c = h.to(self.device), c.to(self.device)
        num_layers = self.lstm.num_layers
        return h[:num_layers], c[:num_layers]

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
            # Pad to max_actions
            pad_width = max_actions - log_probs.shape[1]
            full_log_probs.append(F.pad(log_probs, (0, pad_width), value=float("-inf")))

        return actions, torch.stack(selected_log_probs, dim=-1), torch.stack(entropies, dim=-1), torch.stack(full_log_probs, dim=-1)


class Policy(nn.Module):
    """CNN + Self feature encoder policy for discrete multi-head action space."""

    def __init__(self, env, cnn_channels=128, hidden_size=512, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        max_vec = torch.tensor(
            [9, 1, 1, 10, 3, 254, 1, 1, 235, 8, 9, 250, 29, 1, 1, 8, 1, 1, 6, 3, 1, 2],
            dtype=torch.float32,
        )[None, :, None, None]
        self.register_buffer("max_vec", max_vec.to(self.device))

        self.actor = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in self.action_space.nvec]
        )
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        self.to(self.device)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Converts raw observation tokens into a concatenated self + CNN feature vector."""
        observations = observations.to(self.device)
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]
        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")

        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        x_coords = ((coords_byte >> 4) & 0x0F).long()
        y_coords = (coords_byte & 0x0F).long()
        atr_indices = observations[..., 1].long()
        atr_values = observations[..., 2].float()

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=self.device,
        )

        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )

        batch_idx = torch.arange(B * TT, device=self.device).unsqueeze(-1).expand_as(atr_values)
        box_obs[batch_idx[valid_tokens], atr_indices[valid_tokens], x_coords[valid_tokens], y_coords[valid_tokens]] = atr_values[valid_tokens]

        features = box_obs / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)

        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden: torch.Tensor):
        """Maps hidden features to action logits and value."""
        return [head(hidden) for head in self.actor], self.value(hidden)
