"""Unit tests for the Hierarchical Reasoning Model policy."""

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

from cogames.policy.hrm import HRMPolicy, HRMPolicyNet


class MockEnv:
    """Minimal environment stub mimicking MettaGridEnv attributes."""

    def __init__(self) -> None:
        self.single_observation_space = Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
        self.single_action_space = Discrete(8)


def test_forward_shapes() -> None:
    env = MockEnv()
    net = HRMPolicyNet(env, num_branches=3)
    obs = torch.randint(0, 256, (5, 7, 7, 3), dtype=torch.uint8)

    logits, values = net.forward_eval(obs)

    assert logits.shape == (5, env.single_action_space.n)
    assert values.shape == (5, 1)


def test_reasoning_weights_are_valid_distribution() -> None:
    env = MockEnv()
    net = HRMPolicyNet(env, num_branches=5)
    obs = torch.randint(0, 256, (2, 7, 7, 3), dtype=torch.uint8)

    weights = net.reasoning_weights(obs)

    assert weights.shape == (2, net.num_branches)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(weights >= 0)


def test_agent_policy_produces_valid_actions() -> None:
    env = MockEnv()
    policy = HRMPolicy(env, torch.device("cpu"))
    agent_policy = policy.agent_policy(agent_id=0)
    obs = env.single_observation_space.sample()

    action = agent_policy.step(obs)
    assert env.single_action_space.contains(int(action))
