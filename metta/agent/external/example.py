import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn as nn


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, **kwargs):
        # If no policy provided, create one with the extra kwargs
        if policy is None:
            policy = Policy(env, hidden_size=hidden_size, **kwargs)
        super().__init__(env, policy, input_size, hidden_size)

    def forward(self, observations, state):
        """Forward function for inference with Metta-compatible state handling"""
        # Check if these are token observations [B, M, 3]
        if observations.dim() == 3 and observations.shape[-1] == 3:
            # Token observations path - encode directly
            hidden = self.policy.encode_observations(observations, state=state)
        elif len(observations.shape) == 5:
            # Training path: B, T, H, W, C -> use forward_train
            x = einops.rearrange(observations, "b t h w c -> b t c h w").float()
            return self._forward_train_with_state_conversion(x, state)
        else:
            # Regular inference path: B, H, W, C
            x = einops.rearrange(observations, "b h w c -> b c h w").float() / self.policy.max_vec
            hidden = self.policy.encode_observations(x, state=state)

        # Handle LSTM state
        h, c = state.lstm_h, state.lstm_c
        if h is not None:
            if len(h.shape) == 3:
                h, c = h.squeeze(), c.squeeze()
            assert h.shape[0] == c.shape[0] == observations.shape[0], "LSTM state must be (h, c)"
            lstm_state = (h, c)
        else:
            lstm_state = None

        # LSTM forward pass
        hidden, c = self.cell(hidden, lstm_state)

        # Update state
        state.hidden = hidden
        state.lstm_h = hidden
        state.lstm_c = c

        return self.policy.decode_actions(hidden)

    def _forward_train_with_state_conversion(self, x, state):
        """Helper to handle state conversion for training"""
        if hasattr(state, "lstm_h"):
            # Convert PolicyState to dict for forward_train compatibility
            state_dict = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c, "hidden": getattr(state, "hidden", None)}
            result = self.forward_train(x, state_dict)
            # Update original state
            state.lstm_h = state_dict.get("lstm_h")
            state.lstm_c = state_dict.get("lstm_c")
            state.hidden = state_dict.get("hidden")
            return result
        else:
            return self.forward_train(x, state)


class Policy(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

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

        # Updated max_vec from PufferLib torch.py
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
            ]
        )[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        action_nvec = env.single_action_space.nvec
        self.actor = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in action_nvec]
        )

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return (actions, value), hidden

    def encode_observations(self, observations, state=None):
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1
        if token_observations.dim() != 3:  # hardcoding for shape [B, M, 3]
            TT = token_observations.shape[1]
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"

        token_observations[token_observations == 255] = 0

        # coords_byte contains x and y coordinates in a single byte (first 4 bits are x, last 4 bits are y)
        coords_byte = token_observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range, but we need to make them long for indexing)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]

        atr_indices = token_observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        atr_values = token_observations[..., 2].float()  # Shape: [B_TT, M]

        # In ObservationShaper we permute. Here, we create the observations pre-permuted.
        # We'd like to pre-create this as part of initialization, but we don't know the batch size or time steps at
        # that point.
        box_obs = torch.zeros(
            (B * TT, 22, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )

        batch_indices = torch.arange(B * TT, device=token_observations.device).unsqueeze(-1).expand_as(atr_values)
        valid_tokens = coords_byte != 0xFF

        box_obs[
            batch_indices[valid_tokens],
            atr_indices[valid_tokens],
            x_coord_indices[valid_tokens],
            y_coord_indices[valid_tokens],
        ] = atr_values[valid_tokens]

        observations = box_obs

        features = observations / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)

        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden):
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value
