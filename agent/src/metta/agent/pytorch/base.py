"""Base classes and utilities shared by all PyTorch policy implementations."""

import logging
import math

import torch
import torch.nn as nn
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class LSTMWrapper(nn.Module):
    """Enhanced LSTM wrapper that supports multi-layer LSTMs.

    Based on pufferlib.models.LSTMWrapper but with num_layers support
    to match the YAML implementations which use 2 layers.
    """

    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=2):
        """Initialize LSTM wrapper with configurable number of layers.

        Args:
            env: Environment
            policy: The policy to wrap (must have encode_observations and decode_actions)
            input_size: Input size to LSTM
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers (default 2 to match YAML)
        """
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_continuous = getattr(self.policy, 'is_continuous', False)

        # Create multi-layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

        # Initialize parameters after LSTM creation to match YAML agent
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Match YAML agent initialization
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # Orthogonal initialization

        # Store action conversion tensors (will be set by MettaAgent)
        self.action_index_tensor = None
        self.cum_action_max_params = None


def initialize_action_embeddings(embeddings: nn.Embedding, max_value: float = 0.1):
    """Initialize action embeddings to match YAML ActionEmbedding component.
    
    Args:
        embeddings: The embedding layer to initialize
        max_value: Maximum absolute value to scale embeddings to (default 0.1)
    """
    nn.init.orthogonal_(embeddings.weight)
    with torch.no_grad():
        max_abs_value = torch.max(torch.abs(embeddings.weight))
        embeddings.weight.mul_(max_value / max_abs_value)


def init_bilinear_actor(actor_hidden_dim: int, action_embed_dim: int, dtype=torch.float32):
    """Initialize bilinear actor head to match MettaActorSingleHead.
    
    Args:
        actor_hidden_dim: Hidden dimension size
        action_embed_dim: Action embedding dimension size
        dtype: Data type for parameters
        
    Returns:
        Tuple of (W parameter, bias parameter)
    """
    # Bilinear parameters matching MettaActorSingleHead
    actor_W = nn.Parameter(
        torch.Tensor(1, actor_hidden_dim, action_embed_dim).to(dtype=dtype)
    )
    actor_bias = nn.Parameter(torch.Tensor(1).to(dtype=dtype))
    
    # Kaiming (He) initialization
    bound = 1 / math.sqrt(actor_hidden_dim) if actor_hidden_dim > 0 else 0
    nn.init.uniform_(actor_W, -bound, bound)
    nn.init.uniform_(actor_bias, -bound, bound)
    
    return actor_W, actor_bias


def bilinear_actor_forward(
    actor_features: torch.Tensor,
    action_embeds: torch.Tensor,
    actor_W: torch.Tensor,
    actor_bias: torch.Tensor,
    actor_hidden_dim: int,
    action_embed_dim: int,
) -> torch.Tensor:
    """Perform bilinear interaction for action selection matching MettaActorSingleHead.
    
    Args:
        actor_features: Actor hidden features [batch_size, hidden_dim]
        action_embeds: Action embeddings [batch_size, num_actions, embed_dim]
        actor_W: Bilinear weight parameter
        actor_bias: Bias parameter
        actor_hidden_dim: Hidden dimension size
        action_embed_dim: Action embedding dimension size
        
    Returns:
        Logits tensor [batch_size, num_actions]
    """
    batch_size = actor_features.shape[0]
    num_actions = action_embeds.shape[1]
    
    # Reshape for bilinear calculation
    actor_repeated = actor_features.unsqueeze(1).expand(-1, num_actions, -1)
    actor_reshaped = actor_repeated.reshape(-1, actor_hidden_dim)
    action_embeds_reshaped = action_embeds.reshape(-1, action_embed_dim)
    
    # Perform bilinear operation using einsum (matching MettaActorSingleHead)
    query = torch.einsum("n h, k h e -> n k e", actor_reshaped, actor_W)
    query = torch.tanh(query)
    scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)
    
    biased_scores = scores + actor_bias
    
    # Reshape back to [batch_size, num_actions]
    logits = biased_scores.reshape(batch_size, num_actions)
    
    return logits


def convert_action_to_logit_index(
    flattened_action: torch.Tensor,
    cum_action_max_params: torch.Tensor,
) -> torch.Tensor:
    """Convert flattened actions to logit indices.
    
    NOTE: This function uses the compensating formula that matches ComponentPolicy.
    The cumsum calculation in MettaAgent is technically wrong, but ComponentPolicy
    compensates with this formula. Both must be kept in sync.
    
    Args:
        flattened_action: Actions in [action_type, action_param] format
        cum_action_max_params: Cumulative sum of action max params
        
    Returns:
        Logit indices for the actions
    """
    action_type_numbers = flattened_action[:, 0].long()
    action_params = flattened_action[:, 1].long()
    cumulative_sum = cum_action_max_params[action_type_numbers]
    
    # Match ComponentPolicy's compensating formula
    return action_type_numbers + cumulative_sum + action_params