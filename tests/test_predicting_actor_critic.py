import unittest
import gymnasium as gym
import numpy as np
import torch
from rl_framework.sample_factory.predicting_actor_critic import PredictingActorCritic
from types import SimpleNamespace
from sample_factory.algo.utils.context import SampleFactoryContext
from torch import nn
from sample_factory.model.encoder import Encoder

class MockObsPredictor(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.obs_shape = obs_shape
        self.return_value = torch.zeros(self.obs_shape)

    def forward(self, x):
        return self.return_value

class MockEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.obs_shape = obs_space["obs"].shape
        self.obs_size = np.prod(self.obs_shape)

    def forward(self, obs_dict):
        return torch.zeros((obs_dict["obs"].shape[0], self.obs_size))

    def get_out_size(self) -> int:
        return self.obs_size


class TestPredictingActorCritic(unittest.TestCase):
    def setUp(self):
        sf_context = SampleFactoryContext()
        sf_context.model_factory.register_encoder_factory(MockEncoder)

        self.cfg = SimpleNamespace(
            normalize_input=False,
            obs_subtract_mean=False,
            obs_scale=1,
            normalize_returns=False,
            nonlinearity="relu",
            use_rnn=True,
            rnn_type="gru",
            rnn_size=256,
            rnn_num_layers=2,
            decoder_mlp_layers=[],
            adaptive_stddev=False,
            continuous_tanh_scale=0,
            policy_init_gain=1.0,
            policy_initialization="orthogonal"
            )
        self.obs_space = {
            'obs': gym.spaces.Box(
                low=0, high=255, shape=(1, 2, 3), dtype=np.uint8)
        }
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(4), gym.spaces.Box(-1, 1, (1,))])
        self.actor_critic = PredictingActorCritic(sf_context.model_factory, self.obs_space, self.action_space, self.cfg)
        self.actor_critic.obs_predictor = MockObsPredictor(self.actor_critic._obs_size)

    def test_forward(self):
        batch_size = 23
        obs = torch.tensor([self.obs_space['obs'].sample() for _ in range(batch_size)])
        obs_dict = {'obs': obs}

        errors = torch.rand((batch_size, self.actor_critic._obs_size))
        self.actor_critic.obs_predictor.return_value = \
            obs.view(batch_size, -1).to(torch.float32) + errors

        rnn_states = torch.zeros((batch_size, 2, self.cfg.rnn_size))
        result = self.actor_critic.forward(obs_dict, rnn_states, values_only=False)

        torch.all((result["actions"][:,1] - torch.mean(errors ** 2, dim=1)).eq(0))

if __name__ == '__main__':
    unittest.main()
