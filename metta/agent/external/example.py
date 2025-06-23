import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn as nn
from einops import rearrange


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, cnn_channels=128, input_size=512, hidden_size=512):
        if policy is None:
            policy = Policy(env, cnn_channels=cnn_channels, hidden_size=hidden_size)
        super().__init__(env, policy, input_size, hidden_size)

    def forward(self, observations, state):
        """Forward function for inference. 3x faster than using LSTM directly"""
        if len(observations.shape) == 5:
            # Training path: B, T, H, W, C -> use forward_train
            x = rearrange(observations, "b t h w c -> b t c h w").float()
            return self._forward_train_with_state_conversion(x, state)

        # Inference path: B, H, W, C
        x = rearrange(observations, "b h w c -> b c h w").float() / self.policy.max_vec
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

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(21, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(21, hidden_size // 2)),
            nn.ReLU(),
        )

        # TODO - fix magic numbers!
        # fmt: off
        max_vec = torch.tensor([  1.,   9.,   1.,  30.,   1.,   3., 255.,  26.,   1.,   1.,   1.,   1.,
                1.,  47.,   3.,   3.,   2.,   1.,   1.,   1.,   1.], dtype=torch.float32)[None, :, None, None]
        self.register_buffer("max_vec", max_vec)
        # fmt: on

        action_nvec = env.single_action_space.nvec
        self.actor = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in action_nvec]
        )

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        # self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return (actions, value), hidden

    def encode_observations(self, observations, state=None):
        # observations are already in [batch, channels, height, width] format
        features = observations.float() / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden):
        # hidden = self.layer_norm(hidden)
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value
