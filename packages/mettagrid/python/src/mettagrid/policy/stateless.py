import logging

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
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
        obs_shape: tuple[int, ...],
        action_names: list[str],
    ):
        self._net = net
        self._device = device
        self._num_actions = num_actions
        self._obs_shape = obs_shape
        self._action_names = action_names

    def _obs_to_array(self, obs: MettaGridObservation) -> np.ndarray:
        """Convert AgentObservation tokens to numpy array.

        Mirrors the conversion logic in NimMultiAgentPolicy.step_single.
        """
        num_tokens, token_dim = self._obs_shape
        obs_array = np.full((num_tokens, token_dim), fill_value=255, dtype=np.uint8)
        for idx, token in enumerate(obs.tokens):
            if idx >= num_tokens:
                break
            token_values = token.raw_token
            obs_array[idx, : len(token_values)] = token_values
        return obs_array

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Get action for this agent."""
        # Handle both numpy arrays (from PufferLib) and AgentObservation (from Rollout)
        if isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = self._obs_to_array(obs)

        obs_tensor = torch.tensor(obs_array, device=self._device).unsqueeze(0).float()

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = int(dist.sample().cpu().item())
            return MettaGridAction(name=self._action_names[action_idx])


class StatelessPolicy(MultiAgentPolicy):
    """Stateless feedforward policy."""

    short_names = ["stateless"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: torch.device | str | None = None):
        super().__init__(policy_env_info)
        actions_cfg = policy_env_info.actions
        self._obs_shape = policy_env_info.observation_space.shape
        self._net = StatelessPolicyNet(actions_cfg, self._obs_shape)
        if device is not None:
            self._net = self._net.to(torch.device(device))
        self.num_actions = len(actions_cfg.actions())

    def network(self) -> nn.Module:
        """Return the underlying network for training."""
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Create a Policy instance for a specific agent."""
        current_device = next(self._net.parameters()).device
        return StatelessAgentPolicyImpl(
            self._net,
            current_device,
            self.num_actions,
            self._obs_shape,
            self._policy_env_info.action_names,
        )

    def is_recurrent(self) -> bool:
        return False

    def load_policy_data(self, checkpoint_path: str) -> None:
        """Load network weights from file."""
        device = next(self._net.parameters()).device
        state_dict = torch.load(checkpoint_path, map_location=device)
        self._net.load_state_dict(state_dict)
        self._net = self._net.to(device)

    def save_policy_data(self, checkpoint_path: str) -> None:
        """Save network weights to file."""
        torch.save(self._net.state_dict(), checkpoint_path)
