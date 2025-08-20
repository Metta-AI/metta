import numpy as np
import torch
import torch.nn as nn


class SimpleAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float()
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            logits = self.forward(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            result = actions.numpy()
            if result.ndim == 0:  # Single action
                return np.array([result.item()])
            return result


def create_agent(obs_dim: int, action_dim: int, device: str = "cpu") -> SimpleAgent:
    agent = SimpleAgent(obs_dim, action_dim)
    return agent.to(device)


def compute_policy_loss(
    agent: SimpleAgent, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
) -> torch.Tensor:
    logits = agent(obs)
    log_probs = torch.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Simple REINFORCE loss
    loss = -torch.mean(action_log_probs * rewards)
    return loss
