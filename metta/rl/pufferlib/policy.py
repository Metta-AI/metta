from types import SimpleNamespace

import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)


class Policy(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(34, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(34, hidden_size // 2)),
            nn.ReLU(),
        )

        max_vec = (
            torch.tensor(
                [
                    1,
                    10,
                    30,
                    1,
                    1,
                    255,
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    1,
                    1,
                    1,
                    10,
                    1,
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ]
            )
            .float()
            .view(1, 34, 1, 1)
        )
        self.register_buffer("max_vec", max_vec)

        action_nvec = env.single_action_space.nvec
        self.actor = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in action_nvec]
        )

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        # self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, observations, state=None):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return (actions, value), hidden

    def encode_observations(self, observations, state=None):
        features = observations.permute(0, 3, 1, 2).float() / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden):
        # hidden = self.layer_norm(hidden)
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value


def load_policy(path: str, device: str = "cpu"):
    weights = torch.load(path, map_location=device, weights_only=True)
    num_actions, hidden_size = weights["policy.actor.0.weight"].shape
    num_action_args, _ = weights["policy.actor.1.weight"].shape
    cnn_channels, obs_channels, _, _ = weights["policy.network.0.weight"].shape
    env = SimpleNamespace(
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(shape=[obs_channels, 11, 11]),
    )
    policy = Policy(env, cnn_channels=cnn_channels, hidden_size=hidden_size)
    policy = Recurrent(env, policy)

    policy.load_state_dict(weights)

    return policy
