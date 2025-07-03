"""
Basal Ganglia Reward Shaper

This module implements a two-layer architecture inspired by the basal ganglia:
1. Upstream Network (Reward Generator): Learns to generate dense reward signals
2. Downstream Network (Policy Learner): Standard policy network that receives augmented rewards

The reward generator learns to provide dense, shaped rewards that encourage exploration,
novelty, and mastery, while the policy network learns to maximize these augmented rewards.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class BasalGangliaRewardShaper(LayerBase):
    """
    A custom layer that implements basal ganglia-inspired reward shaping.

    This layer takes observations and generates augmented rewards that are designed
    to encourage exploration, novelty, and mastery. The augmented rewards are then
    used to train the downstream policy network.

    The architecture consists of:
    1. A reward generator network that processes observations and generates dense rewards
    2. A policy network that receives both observations and augmented rewards
    3. A mechanism to combine extrinsic and intrinsic rewards
    """

    def __init__(self, reward_scale=0.1, novelty_weight=0.3, exploration_weight=0.3, mastery_weight=0.4, **cfg):
        super().__init__(**cfg)
        self.reward_scale = reward_scale
        self.novelty_weight = novelty_weight
        self.exploration_weight = exploration_weight
        self.mastery_weight = mastery_weight

        # State tracking for novelty detection
        self.visited_states = set()
        self.state_visit_counts = {}

    def _make_net(self):
        """Create the reward generation network."""
        # The reward generator processes encoded observations to produce augmented rewards
        # This is a simple MLP that learns to generate reward signals
        self._out_tensor_shape = [1]  # Single scalar reward

        net = nn.Sequential(
            nn.Linear(self._in_tensor_shapes[0][0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Bound the reward signal
        )

        # Initialize weights
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

        return net

    def _forward(self, td: TensorDict):
        """Generate augmented rewards based on observations."""
        # Get the encoded observations
        encoded_obs = td[self._sources[0]["name"]]

        # Generate base augmented reward from the network
        base_augmented_reward = self._net(encoded_obs)

        # Add novelty bonus (simplified - in practice this would track state visits)
        novelty_bonus = self._compute_novelty_bonus(encoded_obs)

        # Add exploration bonus (encourage diverse actions)
        exploration_bonus = self._compute_exploration_bonus(encoded_obs)

        # Add mastery bonus (encourage skill development)
        mastery_bonus = self._compute_mastery_bonus(encoded_obs)

        # Combine all reward components
        augmented_reward = (
            base_augmented_reward * self.reward_scale +
            novelty_bonus * self.novelty_weight +
            exploration_bonus * self.exploration_weight +
            mastery_bonus * self.mastery_weight
        )

        # Store reward components in tensor dict for trainer stats collection
        # These will be collected by the trainer's stats system and logged to wandb
        td["basal_ganglia_learned_raw"] = base_augmented_reward
        td["basal_ganglia_novelty_raw"] = novelty_bonus
        td["basal_ganglia_exploration_raw"] = exploration_bonus
        td["basal_ganglia_mastery_raw"] = mastery_bonus

        td["basal_ganglia_learned_weighted"] = base_augmented_reward * self.reward_scale
        td["basal_ganglia_novelty_weighted"] = novelty_bonus * self.novelty_weight
        td["basal_ganglia_exploration_weighted"] = exploration_bonus * self.exploration_weight
        td["basal_ganglia_mastery_weighted"] = mastery_bonus * self.mastery_weight

        td["basal_ganglia_augmented_total"] = augmented_reward

        # Store the augmented reward in the tensor dict
        td[self._name] = augmented_reward

        return td

    def _compute_novelty_bonus(self, encoded_obs):
        """Compute novelty bonus based on state visitation."""
        # Simplified novelty computation - in practice this would track state visits
        # For now, we'll use a simple heuristic based on observation variance
        batch_size = encoded_obs.shape[0]
        novelty_bonus = torch.zeros(batch_size, 1, device=encoded_obs.device)

        # Compute variance across features as a simple novelty measure
        obs_variance = torch.var(encoded_obs, dim=1, keepdim=True)
        novelty_bonus = torch.tanh(obs_variance * 0.1)  # Scale and bound

        return novelty_bonus

    def _compute_exploration_bonus(self, encoded_obs):
        """Compute exploration bonus to encourage diverse behaviors."""
        # Simplified exploration bonus - encourage movement and action diversity
        batch_size = encoded_obs.shape[0]
        exploration_bonus = torch.zeros(batch_size, 1, device=encoded_obs.device)

        # Use the magnitude of the encoded observation as a proxy for exploration
        obs_magnitude = torch.norm(encoded_obs, dim=1, keepdim=True)
        exploration_bonus = torch.tanh(obs_magnitude * 0.01)  # Scale and bound

        return exploration_bonus

    def _compute_mastery_bonus(self, encoded_obs):
        """Compute mastery bonus to encourage skill development."""
        # Simplified mastery bonus - encourage consistent, goal-directed behavior
        batch_size = encoded_obs.shape[0]
        mastery_bonus = torch.zeros(batch_size, 1, device=encoded_obs.device)

        # Use the consistency of the encoded observation as a proxy for mastery
        # Higher consistency (lower variance across time) suggests mastery
        obs_consistency = 1.0 / (1.0 + torch.var(encoded_obs, dim=1, keepdim=True))
        mastery_bonus = torch.tanh(obs_consistency * 0.1)  # Scale and bound

        return mastery_bonus


class BasalGangliaPolicyNetwork(LayerBase):
    """
    Policy network that receives both observations and augmented rewards.

    This network processes observations and augmented rewards to produce
    policy outputs (actions and values). It represents the "cortical" part
    of the basal ganglia analogy.
    """

    def __init__(self, **cfg):
        super().__init__(**cfg)

    def _make_net(self):
        """Create the policy network that processes observations and rewards."""
        # Input size is observation encoding + augmented reward
        input_size = self._in_tensor_shapes[0][0] + self._in_tensor_shapes[1][0]
        self._out_tensor_shape = [128]  # Policy representation size

        net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)  # Final policy representation
        )

        # Initialize weights
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

        return net

    def _forward(self, td: TensorDict):
        """Process observations and augmented rewards to produce policy representation."""
        # Get the observation encoding and augmented reward
        obs_encoding = td[self._sources[0]["name"]]
        augmented_reward = td[self._sources[1]["name"]]

        # Concatenate observations and rewards
        combined_input = torch.cat([obs_encoding, augmented_reward], dim=1)

        # Process through the policy network
        policy_representation = self._net(combined_input)

        # Store the policy representation
        td[self._name] = policy_representation

        return td


class BasalGangliaRewardIntegrator(LayerBase):
    """
    A layer that integrates extrinsic and intrinsic rewards.

    This layer takes the original extrinsic reward from the environment
    and combines it with the intrinsic reward generated by the basal ganglia
    reward shaper. The combined reward is then used for training.
    """

    def __init__(self, extrinsic_weight=1.0, intrinsic_weight=0.1, **cfg):
        super().__init__(**cfg)
        self.extrinsic_weight = extrinsic_weight
        self.intrinsic_weight = intrinsic_weight

    def _make_net(self):
        """This layer doesn't need a neural network, it just combines rewards."""
        self._out_tensor_shape = [1]  # Single scalar combined reward
        return nn.Identity()  # Placeholder, not actually used

    def _forward(self, td: TensorDict):
        """Combine extrinsic and intrinsic rewards."""
        # Get the extrinsic reward (from environment) and intrinsic reward (from basal ganglia)
        extrinsic_reward = td[self._sources[0]["name"]]
        intrinsic_reward = td[self._sources[1]["name"]]

        # Combine the rewards
        combined_reward = (
            extrinsic_reward * self.extrinsic_weight +
            intrinsic_reward * self.intrinsic_weight
        )

        # Store the combined reward
        td[self._name] = combined_reward

        return td


class GridObservationShaper(LayerBase):
    """
    A simple observation shaper for grid observations.

    This layer takes grid observations of shape [batch, channels, width, height]
    and passes them through unchanged, setting up the proper tensor shapes.
    """

    def __init__(self, obs_shape, obs_width, obs_height, feature_normalizations, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)
        self.out_width = obs_width
        self.out_height = obs_height
        self.num_layers = max(feature_normalizations.keys()) + 1
        self._out_tensor_shape = [self.num_layers, self.out_width, self.out_height]

    def _make_net(self):
        """Create the network (identity function for grid observations)."""
        return nn.Identity()

    def _forward(self, td: TensorDict):
        """Process grid observations."""
        grid_obs = td["x"]

        # Ensure the observation has the right shape
        B = grid_obs.shape[0]
        TT = 1
        if grid_obs.dim() != 4:  # [B, C, W, H]
            TT = grid_obs.shape[1]
            grid_obs = grid_obs.view(B * TT, *grid_obs.shape[2:])

        td["_TT_"] = TT
        td["_batch_size_"] = B
        td["_BxTT_"] = B * TT
        td[self._name] = grid_obs

        return td
