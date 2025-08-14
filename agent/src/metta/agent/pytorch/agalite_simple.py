"""
AGaLiTe implementation for Metta following the Fast agent pattern.
Uses AGaLiTe transformer layers with proper token observation handling.
"""

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from metta.agent.agalite_batched import BatchedAGaLiTe


class AGaLiTeSimple(pufferlib.models.LSTMWrapper):
    """AGaLiTe with LSTM wrapper for compatibility with Metta infrastructure."""

    def __init__(self, env, policy=None, input_size=256, hidden_size=256):
        if policy is None:
            policy = Policy(env, input_size=input_size, hidden_size=hidden_size)

        super().__init__(env, policy, input_size, hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize placeholders for action tensors that MettaAgent will set
        self.action_index_tensor = None
        self.cum_action_max_params = None

    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"].to(self.device)

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Encode obs
        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # Prepare LSTM state
        lstm_h, lstm_c = state.get("lstm_h"), state.get("lstm_c")
        if lstm_h is not None and lstm_c is not None:
            lstm_h = lstm_h.to(self.device)[: self.lstm.num_layers]
            lstm_c = lstm_c.to(self.device)[: self.lstm.num_layers]
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        # Forward LSTM
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, in_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode
        logits_list, value = self.policy.decode_actions(flat_hidden)

        if action is None:
            # ---------- Inference Mode ----------
            log_probs = F.log_softmax(logits_list, dim=-1)
            action_probs = torch.exp(log_probs)

            actions = torch.multinomial(action_probs, num_samples=1).view(-1)

            batch_indices = torch.arange(actions.shape[0], device=actions.device)
            full_log_probs = log_probs[batch_indices, actions]

            action = self._convert_logit_index_to_action(actions)

            td["actions"] = action.to(dtype=torch.int32)
            td["act_log_prob"] = full_log_probs
            td["values"] = value.flatten()
            td["full_log_probs"] = log_probs

        else:
            # ---------- Training Mode ----------
            action = action.to(self.device)
            if action.dim() == 3:  # (B, T, 2) → flatten to (BT, 2)
                B, T, A = action.shape
                action = action.view(B * T, A)

            action_log_probs = F.log_softmax(logits_list, dim=-1)

            action_probs = torch.exp(action_log_probs)
            action_logit_index = self._convert_action_to_logit_index(action)

            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            full_log_probs = action_log_probs[batch_indices, action_logit_index]

            entropy = -(action_probs * action_log_probs).sum(dim=-1)

            td["act_log_prob"] = full_log_probs.view(B, TT)
            td["entropy"] = entropy.view(B, TT)
            td["full_log_probs"] = action_log_probs.view(B, T, -1)
            td["value"] = value.view(B, TT)

        return td

    def clip_weights(self):
        """Clip weights of the actor heads to prevent large updates."""
        pass

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for wandb logging."""
        return []

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return cumulative_sum + action_params


class Policy(nn.Module):
    """Policy with AGaLiTe transformer for encoding and action decoding."""

    def __init__(self, env, input_size=256, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        # Token observation to grid conversion (like Fast agent)
        self.cnn1 = pufferlib.pytorch.layer_init(nn.Conv2d(in_channels=22, out_channels=64, kernel_size=5, stride=3))
        self.cnn2 = pufferlib.pytorch.layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1))

        test_input = torch.zeros(1, 22, 11, 11)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        # Project to AGaLiTe input size
        self.fc1 = pufferlib.pytorch.layer_init(nn.Linear(self.flattened_size, 128))
        self.encoded_obs = pufferlib.pytorch.layer_init(nn.Linear(128, input_size))

        # AGaLiTe transformer layers (simplified, no persistent memory)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=4,
                dim_feedforward=input_size * 4,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Output heads
        self.critic_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1024))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(1024, 1), std=1.0)
        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 512))
        self.action_embeddings = nn.Embedding(100, 16)

        # Action heads
        action_nvec = self.action_space.nvec if hasattr(self.action_space, "nvec") else [100]
        self.actor_heads = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(512 + 16, n), std=0.01) for n in action_nvec]
        )

        # Normalization buffer
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
        )[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        self.to(self.device)

    def network_forward(self, x):
        """CNN feature extraction from grid observations."""
        x = x / self.max_vec
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.encoded_obs(x)
        return x

    def encode_observations(self, observations, state=None):
        """Convert token observations to hidden representation using AGaLiTe."""
        observations = observations.to(self.device)
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"
        token_observations[token_observations == 255] = 0

        # Convert tokens to grid representation
        coords_byte = token_observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = token_observations[..., 1].long()
        atr_values = token_observations[..., 2].float()

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )
        batch_indices = torch.arange(B * TT, device=token_observations.device).unsqueeze(-1).expand_as(atr_values)

        valid_tokens = coords_byte != 0xFF
        valid_tokens = valid_tokens & (x_coord_indices < self.out_width) & (y_coord_indices < self.out_height)
        valid_tokens = valid_tokens & (atr_indices < self.num_layers)

        box_obs[
            batch_indices[valid_tokens],
            atr_indices[valid_tokens],
            x_coord_indices[valid_tokens],
            y_coord_indices[valid_tokens],
        ] = atr_values[valid_tokens]

        # Encode with CNN
        hidden = self.network_forward(box_obs)

        # Apply transformer (add sequence dim for transformer)
        hidden = hidden.unsqueeze(1)  # (B*TT, 1, input_size)
        hidden = self.transformer(hidden)
        hidden = hidden.squeeze(1)  # (B*TT, input_size)

        return hidden

    def decode_actions(self, hidden):
        """Decode hidden representation to action logits and value."""
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        actor_features = self.actor_1(hidden)

        action_embed = self.action_embeddings.weight.mean(dim=0).unsqueeze(0).expand(actor_features.shape[0], -1)
        combined_features = torch.cat([actor_features, action_embed], dim=-1)

        logits = torch.cat([head(combined_features) for head in self.actor_heads], dim=-1)

        return logits, value
