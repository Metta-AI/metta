import numpy as np
import torch
import torch.nn as nn


class ActorCriticAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()

        # Shared feature extractor
        self.shared_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_size, action_dim)

        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor):
        features = self.shared_network(obs)
        action_logits = self.actor_head(features)
        values = self.critic_head(features)
        return action_logits, values

    def get_action_and_value(self, obs: torch.Tensor):
        """Get action probabilities and state values."""
        action_logits, values = self.forward(obs)
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs, values

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample an action for environment interaction."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float()
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            action_logits, _ = self.forward(obs_tensor)
            probs = torch.softmax(action_logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            result = actions.numpy()
            if result.ndim == 0:  # Single action
                return np.array([result.item()])
            return result
