import logging

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action as MettaGridAction
from mettagrid.simulator import AgentObservation as MettaGridObservation

logger = logging.getLogger("mettagrid.policy.stateless_policy")


class StatelessPolicyNet(torch.nn.Module):
    """Stateless feedforward policy network."""

    def __init__(self, actions_cfg: ActionsConfig, obs_shape: tuple):
        super().__init__()
        self.hidden_size = 128
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(
                torch.nn.Linear(np.prod(obs_shape).item(), self.hidden_size),
            ),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, self.hidden_size)),
        )

        self.num_actions = len(actions_cfg.actions())

        self.action_head = torch.nn.Linear(self.hidden_size, self.num_actions)
        self.value_head = torch.nn.Linear(self.hidden_size, 1)

    def forward_eval(self, observations, state=None):
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1).float() / 255.0
        hidden = self.net(observations)
        logits = self.action_head(hidden)

        values = self.value_head(hidden)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class StatelessAgentPolicyImpl(AgentPolicy):
    """Per-agent policy that uses the shared feedforward network."""

    def __init__(
        self,
        net: StatelessPolicyNet,
        device: torch.device,
        num_actions: int,
        expected_num_tokens: int,
        policy_env_info: PolicyEnvInterface,
    ):
        super().__init__(policy_env_info)
        self._net = net
        self._device = device
        self._num_actions = num_actions
        self._expected_num_tokens = expected_num_tokens

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Get action for this agent."""
        # Convert AgentObservation to numpy array format: (num_tokens, 3)
        # Each token is [coords_byte, feature_id, value]
        tokens = []
        for token in obs.tokens:
            col, row = token.location
            coords_byte = ((col & 0x0F) << 4) | (row & 0x0F)
            feature_id = token.feature.id
            value = token.value
            tokens.append([coords_byte, feature_id, value])

        # Pad to expected number of tokens (padding uses coords_byte=255)
        num_tokens = len(tokens)
        if num_tokens < self._expected_num_tokens:
            padding_needed = self._expected_num_tokens - num_tokens
            tokens.extend([[255, 0, 0]] * padding_needed)

        # Convert to numpy array and then to tensor
        obs_array = np.array(tokens, dtype=np.uint8)
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(self._device).float()

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            sampled_action = dist.sample().cpu().item()
            # Convert action index to Action object
            action = list(self._policy_env_info.actions.actions())[sampled_action]
            return action


class StatelessPolicy(TrainablePolicy):
    """Stateless feedforward policy."""

    def __init__(
        self, actions_cfg: ActionsConfig, obs_shape: tuple, device: torch.device, policy_env_info: PolicyEnvInterface
    ):
        super().__init__(policy_env_info)
        self._net = StatelessPolicyNet(actions_cfg, obs_shape).to(device)
        self._device = device
        self.num_actions = len(actions_cfg.actions())
        self._expected_num_tokens = obs_shape[0]

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Create a Policy instance for a specific agent."""
        return StatelessAgentPolicyImpl(
            self._net, self._device, self.num_actions, self._expected_num_tokens, self._policy_env_info
        )

    def is_recurrent(self) -> bool:
        return False

    def load_policy_data(self, checkpoint_path: str) -> None:
        self._net.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        self._net = self._net.to(self._device)

    def save_policy_data(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
