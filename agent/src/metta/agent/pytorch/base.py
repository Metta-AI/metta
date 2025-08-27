import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMWrapper(nn.Module):
    """Enhanced LSTM wrapper that supports multi-layer LSTMs with automatic memory management.

    This base class provides:
    - Multi-layer LSTM support
    - Automatic state detachment to prevent gradient accumulation
    - Per-environment state tracking
    - Episode boundary reset handling
    - Memory management interface

    All LSTM-based policies inherit these critical features automatically."""

    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=2):
        """Initialize LSTM wrapper with configurable number of layers."""
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_continuous = getattr(self.policy, "is_continuous", False)

        # Create multi-layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

        # Initialize parameters after LSTM creation to match ComponentPolicy
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Match ComponentPolicy initialization
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # Match ComponentPolicy initialization

        # Store action conversion tensors (will be set by MettaAgent)
        self.action_index_tensor = None
        self.cum_action_max_params = None

        # LSTM memory management - critical for stable training
        self.lstm_h = {}  # Hidden states per environment
        self.lstm_c = {}  # Cell states per environment

    # ============================================================================
    # Memory Management Methods - Available to all LSTM policies
    # ============================================================================

    def has_memory(self):
        """Indicate that this policy has memory (LSTM states)."""
        return True

    def get_memory(self):
        """Get current LSTM memory states for checkpointing."""
        return self.lstm_h, self.lstm_c

    def set_memory(self, memory):
        """Set LSTM memory states from checkpoint."""
        self.lstm_h, self.lstm_c = memory[0], memory[1]

    def reset_memory(self):
        """Reset all LSTM memory states."""
        self.lstm_h.clear()
        self.lstm_c.clear()

    def reset_env_memory(self, env_id):
        """Reset LSTM memory for a specific environment."""
        if env_id in self.lstm_h:
            del self.lstm_h[env_id]
        if env_id in self.lstm_c:
            del self.lstm_c[env_id]

    def _manage_lstm_state(self, td, B, TT, device):
        """Manage LSTM state with automatic reset and detachment.

        This method handles:
        - Per-environment state tracking
        - Episode boundary resets
        - State initialization
        - Gradient detachment

        Args:
            td: TensorDict containing environment info
            B: Batch size
            TT: Time steps
            device: Device for tensor allocation

        Returns:
            tuple: (lstm_h, lstm_c, env_id) ready for LSTM forward pass
        """
        # Get environment ID for state tracking
        training_env_id_start = td.get("training_env_id_start", None)
        if training_env_id_start is None:
            training_env_id_start = 0
        else:
            training_env_id_start = training_env_id_start[0].item()

        # Prepare LSTM state with proper memory management
        if training_env_id_start in self.lstm_h and training_env_id_start in self.lstm_c:
            lstm_h = self.lstm_h[training_env_id_start]
            lstm_c = self.lstm_c[training_env_id_start]

            # Reset hidden state if episode is done or truncated
            dones = td.get("dones", None)
            truncateds = td.get("truncateds", None)
            if dones is not None and truncateds is not None:
                reset_mask = (dones.bool() | truncateds.bool()).view(1, -1, 1)
                lstm_h = lstm_h.masked_fill(reset_mask, 0)
                lstm_c = lstm_c.masked_fill(reset_mask, 0)
        else:
            # Initialize new hidden states
            lstm_h = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            lstm_c = torch.zeros(self.num_layers, B, self.hidden_size, device=device)

        return lstm_h, lstm_c, training_env_id_start

    def _store_lstm_state(self, lstm_h, lstm_c, env_id):
        """Store LSTM state with automatic detachment to prevent gradient accumulation.

        CRITICAL: The detach() call here prevents gradients from flowing backward
        through time infinitely, which would cause memory leaks and training instability.

        Args:
            lstm_h: LSTM hidden state to store
            lstm_c: LSTM cell state to store
            env_id: Environment ID for state tracking
        """
        self.lstm_h[env_id] = lstm_h.detach()
        self.lstm_c[env_id] = lstm_c.detach()


def initialize_action_embeddings(embeddings: nn.Embedding, max_value: float = 0.1):
    """Initialize action embeddings to match YAML ActionEmbedding component."""
    nn.init.orthogonal_(embeddings.weight)
    with torch.no_grad():
        max_abs_value = torch.max(torch.abs(embeddings.weight))
        embeddings.weight.mul_(max_value / max_abs_value)


def init_bilinear_actor(actor_hidden_dim: int, action_embed_dim: int, dtype=torch.float32):
    """Initialize bilinear actor head to match MettaActorSingleHead."""
    # Bilinear parameters matching MettaActorSingleHead
    actor_W = nn.Parameter(torch.Tensor(1, actor_hidden_dim, action_embed_dim).to(dtype=dtype))
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
    """Perform bilinear interaction for action selection matching MettaActorSingleHead."""
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
    """Convert flattened actions to logit indices for MultiDiscrete action spaces."""
    action_type_numbers = flattened_action[:, 0].long()
    action_params = flattened_action[:, 1].long()
    cumulative_sum = cum_action_max_params[action_type_numbers]

    # Formula for MultiDiscrete action space conversion
    return action_type_numbers + cumulative_sum + action_params
