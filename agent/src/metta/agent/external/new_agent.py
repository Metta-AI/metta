import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from metta.agent.external.example import Recurrent


class PolicyBase(nn.Module, ABC):
    """Base class for all policies with standardized interface."""

    def __init__(self, name: str = "BasePolicy"):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(
        self,
        agent: "MettaAgent",
        obs: Dict[str, torch.Tensor],
        state: Optional[Any] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Execute policy forward pass."""
        pass


@dataclass
class AgentSpec:
    """Specification for agent configuration and validation."""

    obs_space: gym.Space
    action_space: gym.Space
    obs_width: int
    obs_height: int
    feature_normalizations: Dict[str, Any]
    global_features: Dict[str, Any]
    device: str
    required_components: List[str]
    optional_components: List[str] = None


class ObsEncoder(nn.Module):
    """Enhanced observation encoder with multiple input support."""

    def __init__(self, input_dim=10, output_dim=32, dropout_rate=0.0, activation="relu", **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            self._get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            self._get_activation(activation),
        )

    def _get_activation(self, activation):
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU(), "leaky_relu": nn.LeakyReLU()}
        return activations.get(activation, nn.ReLU())

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "grid_obs" not in obs:
            raise KeyError("Expected 'grid_obs' in observation dict")
        return self.layers(obs["grid_obs"])


class RecurrentCore(nn.Module):
    """Enhanced recurrent core with multiple architectures."""

    def __init__(self, input_dim=32, hidden_size=32, num_layers=1, rnn_type="lstm", dropout=0.0, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden state."""
        if self.rnn_type == "lstm":
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h, c)
        else:  # GRU
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, x: torch.Tensor, state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        if state is None:
            state = self.init_hidden(x.size(0), x.device)

        x, new_state = self.rnn(x.unsqueeze(1), state)
        return x.squeeze(1), new_state


class ActionHead(nn.Module):
    """Enhanced action head with multiple output distributions."""

    def __init__(self, hidden_size=32, action_space=None, temperature=1.0, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_space = action_space
        self.temperature = temperature

        if isinstance(action_space, gym.spaces.Discrete):
            self.output_dim = action_space.n
            self.action_type = "discrete"
        elif isinstance(action_space, gym.spaces.Box):
            self.output_dim = action_space.shape[0] * 2  # mean and std
            self.action_type = "continuous"
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.layers(x)

        if self.action_type == "discrete":
            return logits / self.temperature
        else:
            # For continuous actions, return mean and log_std
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            return torch.cat([mean, log_std], dim=-1)


class ValueHead(nn.Module):
    """Enhanced value head with multiple value types support."""

    def __init__(self, hidden_size=32, num_values=1, **kwargs):
        super().__init__()
        self.num_values = num_values
        self.layers = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MettaAgent(nn.Module):
    """Enhanced MettaAgent with better error handling and introspection."""

    def __init__(
        self,
        components: nn.ModuleDict,
        agent_spec: AgentSpec,
        config: DictConfig,
        policy: Optional[PolicyBase] = None,
    ):
        super().__init__()
        self.components = components
        self.agent_spec = agent_spec
        self.config = config
        self.policy = policy
        self.device = agent_spec.device

        # Validate required components
        self._validate_components()

        # Move to device
        self.to(self.device)

    def _validate_components(self):
        """Validate that all required components are present."""
        missing = []
        for req_comp in self.agent_spec.required_components:
            if req_comp not in self.components:
                missing.append(req_comp)

        if missing:
            raise ValueError(f"Missing required components: {missing}")

    def set_policy(self, policy: PolicyBase):
        """Set or change the agent's policy."""
        if not isinstance(policy, PolicyBase):
            raise TypeError("Policy must inherit from PolicyBase")
        self.policy = policy

    def forward(
        self, obs: Dict[str, torch.Tensor], state: Optional[Any] = None, action: Optional[torch.Tensor] = None
    ) -> Tuple:
        if self.policy is None:
            raise RuntimeError("No policy set. Use set_policy() first.")
        return self.policy(self, obs, state, action)


class MettaAgentBuilder:
    """Enhanced builder with better validation and error handling."""

    def __init__(self, config: Union[Dict, DictConfig, str], agent_spec: AgentSpec):
        if isinstance(config, str):
            self.cfg = OmegaConf.load(config)
        else:
            self.cfg = OmegaConf.create(config)

        self.agent_spec = agent_spec
        self.logger = logging.getLogger(__name__)

    def build(self, policy: Optional[Union[PolicyBase, DictConfig, Dict]] = None) -> MettaAgent:
        """Build a MettaAgent instance."""
        try:
            components = self._build_components()

            if policy is None:
                policy = self._build_default_policy()
            elif not isinstance(policy, PolicyBase):
                policy = self._instantiate_policy(policy)

            agent = MettaAgent(components=components, agent_spec=self.agent_spec, config=self.cfg, policy=policy)

            self.logger.info(f"Built MettaAgent with {len(components)} components")
            return agent

        except Exception as e:
            self.logger.error(f"Failed to build MettaAgent: {e}")
            raise

    def _build_components(self) -> nn.ModuleDict:
        """Build all components from configuration."""
        components = nn.ModuleDict()

        for name, comp_cfg in self.cfg.components.items():
            try:
                # Merge with agent spec attributes
                full_cfg = dict(comp_cfg)
                full_cfg.update(
                    {
                        "obs_space": self.agent_spec.obs_space,
                        "action_space": self.agent_spec.action_space,
                        "obs_width": self.agent_spec.obs_width,
                        "obs_height": self.agent_spec.obs_height,
                        "device": self.agent_spec.device,
                        "feature_normalizations": self.agent_spec.feature_normalizations,
                        "global_features": self.agent_spec.global_features,
                    }
                )

                if hasattr(self.cfg, "hidden_size"):
                    full_cfg["hidden_size"] = self.cfg.hidden_size

                component = instantiate(full_cfg)
                components[name] = component
                self.logger.debug(f"Built component: {name}")

            except Exception as e:
                self.logger.error(f"Failed to build component {name}: {e}")
                raise

        return components

    def _build_default_policy(self) -> PolicyBase:
        """Build default policy if none provided."""
        return ComponentBasedPolicy()

    def _instantiate_policy(self, policy_config: Union[DictConfig, Dict]) -> PolicyBase:
        """Instantiate policy from configuration."""
        if isinstance(policy_config, (DictConfig, dict)) and "_target_" in policy_config:
            return instantiate(policy_config)
        else:
            raise TypeError("Policy config must have '_target_' field")


class ComponentBasedPolicy(PolicyBase):
    """Policy that uses modular components (Approach 1)."""

    def __init__(self, name: str = "ComponentBased"):
        super().__init__(name)

    def forward(
        self,
        agent: MettaAgent,
        obs: Dict[str, torch.Tensor],
        state: Optional[Any] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple:
        # Encode observations
        x = agent.components["obs_encoder"](obs)

        # Pass through recurrent core
        x, new_state = agent.components["recurrent_core"](x, state)

        # Get action logits and value
        action_output = agent.components["action_head"](x)
        value = agent.components["value_head"](x).squeeze(-1)

        # Handle different action spaces
        if isinstance(agent.agent_spec.action_space, gym.spaces.Discrete):
            dist = torch.distributions.Categorical(logits=action_output)
        else:
            # Continuous actions - split into mean and std
            mean, log_std = torch.chunk(action_output, 2, dim=-1)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        if len(log_prob.shape) > 1:  # Multi-dimensional continuous actions
            log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy()
        if len(entropy.shape) > 1:  # Multi-dimensional continuous actions
            entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy, value, new_state


class MonolithicPolicy(PolicyBase):
    """Monolithic PyTorch policy with all components in one module (Approach 2)."""

    def __init__(
        self,
        obs_dim: int,
        action_space: gym.Space,
        hidden_size: int = 64,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        name: str = "Monolithic",
    ):
        super().__init__(name)
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size * 2), nn.ReLU(), nn.Linear(hidden_size * 2, hidden_size), nn.ReLU()
        )

        # Recurrent core
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Action head
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_space.n)
            )
            self.action_type = "discrete"
        elif isinstance(action_space, gym.spaces.Box):
            self.action_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_space.shape[0] * 2),  # mean and log_std
            )
            self.action_type = "continuous"
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        # Value head
        self.value_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden state."""
        if self.rnn_type == "lstm":
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h, c)
        else:  # GRU
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(
        self,
        agent: MettaAgent,
        obs: Dict[str, torch.Tensor],
        state: Optional[Any] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple:
        # Extract grid observation
        grid_obs = obs["grid_obs"]
        batch_size = grid_obs.size(0)

        # Encode observations
        x = self.obs_encoder(grid_obs)

        # Initialize state if needed
        if state is None:
            state = self.init_hidden(batch_size, grid_obs.device)

        # Pass through RNN
        x, new_state = self.rnn(x.unsqueeze(1), state)
        x = x.squeeze(1)

        # Get action logits and value
        action_output = self.action_head(x)
        value = self.value_head(x).squeeze(-1)

        # Handle different action spaces
        if self.action_type == "discrete":
            dist = torch.distributions.Categorical(logits=action_output)
        else:
            # Continuous actions - split into mean and std
            mean, log_std = torch.chunk(action_output, 2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        if len(log_prob.shape) > 1:  # Multi-dimensional continuous actions
            log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy()
        if len(entropy.shape) > 1:  # Multi-dimensional continuous actions
            entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy, value, new_state


class PufferlibRecurrentPolicy(PolicyBase):
    def __init__(self, env):
        super().__init__("PufferlibRecurrent")
        self.action_space = env.single_action_space

        if isinstance(self.action_space, spaces.Discrete):
            # treat Discrete(N) as a 1-D MultiDiscrete([N])
            self.action_nvec = np.array([self.action_space.n], dtype=int)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.action_nvec = self.action_space.nvec
        else:
            raise NotImplementedError(f"Unsupported action space: {type(self.action_space)}")

        self.recurrent_agent = Recurrent(env)
        self.recurrent_agent.device = "cpu"
        self.recurrent_agent.eval()  # Optional: set eval mode

    def forward(
        self,
        agent: MettaAgent,
        obs: Dict[str, torch.Tensor],
        state: Optional[dict] = None,
        action: Optional[torch.Tensor] = None,
    ):
        """
        MettaAgent's obs -> Tensor: (B, M, 3) or (B, TT, M, 3)
        You must convert the dictionary to the tensor format expected by the pufferlib agent.
        """
        observations = obs["obs"]  # Use correct key based on env
        output = self.recurrent_agent(observations, state=state, action=action)
        return output


def create_component_based_agent():
    """Create an example MettaAgent using component-based approach (Approach 1)."""

    # Define agent specification
    agent_spec = AgentSpec(
        obs_space=gym.spaces.Dict(
            {
                "grid_obs": gym.spaces.Box(low=0, high=1, shape=(10,), dtype=float),
                "global_vars": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,), dtype=int),
            }
        ),
        action_space=gym.spaces.Discrete(5),
        obs_width=5,
        obs_height=5,
        feature_normalizations={},
        global_features={},
        device="cpu",
        required_components=["obs_encoder", "recurrent_core", "action_head", "value_head"],
    )

    # Define configuration
    config = {
        "hidden_size": 64,
        "clip_range": 0.2,
        "components": {
            "obs_encoder": {"_target_": "__main__.ObsEncoder", "input_dim": 10, "output_dim": 64, "dropout_rate": 0.1},
            "recurrent_core": {
                "_target_": "__main__.RecurrentCore",
                "input_dim": 64,
                "hidden_size": 64,
                "num_layers": 2,
                "rnn_type": "lstm",
            },
            "action_head": {"_target_": "__main__.ActionHead", "hidden_size": 64},
            "value_head": {"_target_": "__main__.ValueHead", "hidden_size": 64},
        },
    }

    # Build agent
    builder = MettaAgentBuilder(config, agent_spec)
    policy = ComponentBasedPolicy()
    agent = builder.build(policy=policy)

    return agent, agent_spec


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    import numpy as np
    import torch
    from gym import spaces

    def env_creator(name: str = None):
        """
        Factory function for creating the MettaGrid environment.
        The `name` parameter is ignored but kept for API compatibility.
        """
        return MettaGridEnv()

    class MettaGridEnv:
        """
        A simple grid-based environment for pufferlib agents.
        Observation space: 11x11 grid with a single channel (float32).
        Action space: 4 discrete actions (up, down, left, right).
        """

        def __init__(self):
            # Single-agent observation & action spaces
            self.single_observation_space = spaces.Box(low=0.0, high=1.0, shape=(11, 11, 3), dtype=np.float32)
            self.single_action_space = spaces.MultiDiscrete([4])

            # For compatibility with pufferlib VecEnv wrappers
            self.observation_space = self.single_observation_space
            self.action_space = self.single_action_space

            # Internal state
            self.grid = np.zeros((11, 11, 3), dtype=np.float32)
            self.agent_pos = [5, 5]

        def reset(self):
            """
            Reset the environment to an initial state.
            Returns:
                obs (np.ndarray): Observation of shape (11, 11, 1).
                info (dict): Empty info dict.
            """
            self.grid.fill(0)
            self.agent_pos = [5, 5]
            self._place_agent()
            obs = self._get_obs()
            return obs, {}

        def step(self, action: int):
            """
            Take an action in the environment.
            Args:
                action (int): Discrete action (0=up, 1=down, 2=left, 3=right).
            Returns:
                (obs, info), reward, done, { }
            """
            # Clear old agent position
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 0.0

            # Move agent
            if action == 0 and self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
            elif action == 1 and self.agent_pos[0] < 10:
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
            elif action == 3 and self.agent_pos[1] < 10:
                self.agent_pos[1] += 1

            # Place agent in new position
            self._place_agent()

            # Observations
            obs = self._get_obs()
            reward = 0.0
            done = False
            info = {}
            return (obs, info), reward, done, {}

        def _place_agent(self):
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 1.0

        def _get_obs(self) -> np.ndarray:
            # Expand grid to have channel axis
            return self.grid[..., None]

        def render(self, mode="human"):
            # Simple text render
            grid = np.array(self.grid, copy=True)
            grid[self.agent_pos[0], self.agent_pos[1]] = 9
            print(grid)

        def close(self):
            pass

    env = env_creator()

    agent_spec = AgentSpec(
        obs_space=env.single_observation_space,
        action_space=env.single_action_space,
        obs_width=11,
        obs_height=11,
        feature_normalizations={},
        global_features={},
        device="cuda" if torch.cuda.is_available() else "cpu",
        required_components=[],
        optional_components=[],
    )

    config = {
        "hidden_size": 512,
        "components": {},
    }

    builder = MettaAgentBuilder(config, agent_spec)
    policy = PufferlibRecurrentPolicy(env)
    agent = builder.build(policy=policy)

    # Wrap obs into expected format
    obs, _ = env.reset()
    print("Shape:", obs.shape)
    obs_tensor = torch.tensor(obs).squeeze(3)
    print(obs_tensor.shape)

    obs_dict = {
        "obs": obs_tensor.to(agent_spec.device),
        "components": {},
        "global_features": torch.zeros(1, 0).to(agent_spec.device),
    }

    output = agent(obs_dict)
    actions, selected_action_log_probs, entropy, value, logits_list = output

    print("actions shape:", actions.shape)
    print("selected_action_log_probs shape:", selected_action_log_probs.shape)
    print("entropy shape:", entropy.shape)
    print("value shape:", value.shape)
