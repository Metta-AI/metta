import logging
from typing import Tuple

import einops
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.pytorch.base import LSTMWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin
from pufferlib.pytorch import layer_init as init_layer

from metta.agent.modules.agalite_batched import BatchedAGaLiTe

logger = logging.getLogger(__name__)


class AgaliteHybrid(PyTorchAgentMixin, LSTMWrapper):
    """Hybrid AGaLiTe-LSTM architecture for efficient RL training.

    This uses AGaLiTe's sophisticated attention-based observation encoding
    combined with LSTM for temporal processing. This hybrid approach provides:
    - AGaLiTe's powerful observation processing with linear attention
    - LSTM's efficient O(1) temporal state updates
    - Full compatibility with Metta's training infrastructure via mixin
    - Automatic gradient detachment and proper state management

    Note: This is not the pure AGaLiTe transformer from the paper, but a
    practical hybrid that achieves better training efficiency."""

    def __init__(self, env, policy=None, input_size=256, hidden_size=256, num_layers=2, **kwargs):
        """Initialize with mixin support for configuration parameters.
        
        Args:
            env: Environment
            policy: Optional inner policy 
            input_size: LSTM input size
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            **kwargs: Configuration parameters (clip_range, analyze_weights_interval, etc.)
        """
        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)
        
        if policy is None:
            policy = AgalitePolicy(env, input_size=input_size, hidden_size=hidden_size)

        # Use enhanced LSTMWrapper with proper state management
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass with enhanced state management."""
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Determine dimensions from observations
        if observations.dim() == 4:  # Training
            B = observations.shape[0]
            TT = observations.shape[1]
            # Reshape TD for training if needed
            if td.batch_dims > 1:
                td = td.reshape(B * TT)
        else:  # Inference
            B = observations.shape[0]
            TT = 1

        # Set critical TensorDict fields using mixin
        self.set_tensordict_fields(td, observations)

        # Encode observations through policy
        hidden = self.policy.encode_observations(observations, state)

        # Use enhanced LSTMWrapper's state management (includes automatic detachment!)
        lstm_h, lstm_c, env_id = self._manage_lstm_state(td, B, TT, observations.device)
        lstm_state = (lstm_h, lstm_c)

        # LSTM forward pass
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, hidden)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)

        # Store state with automatic detachment to prevent gradient accumulation
        self._store_lstm_state(new_lstm_h, new_lstm_c, env_id)

        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode actions and value
        logits, value = self.policy.decode_actions(flat_hidden)

        # Use mixin for mode-specific processing
        if action is None:
            # Mixin handles inference mode properly
            td = self.forward_inference(td, logits, value)
        else:
            # Mixin handles training mode with proper reshaping
            td = self.forward_training(td, action, logits, value)

        return td


class AgalitePolicy(nn.Module):
    """Inner policy using AGaLiTe architecture."""

    def __init__(self, env, input_size=256, hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = False  # Required by PufferLib
        self.action_space = env.single_action_space
        # Observation parameters
        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        # Create observation encoders
        self.cnn_encoder = nn.Sequential(
            init_layer(nn.Conv2d(self.num_layers, 128, 5, stride=3)),
            nn.ReLU(),
            init_layer(nn.Conv2d(128, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(128, input_size // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            init_layer(nn.Linear(self.num_layers, input_size // 2)),
            nn.ReLU(),
        )

        # AGaLiTe core (without LSTM since PufferLib wrapper handles that)
        self.agalite_core = BatchedAGaLiTe(
            n_layers=2,  # Reduced for efficiency
            d_model=input_size,
            d_head=64,
            d_ffc=hidden_size * 2,
            n_heads=4,
            eta=4,
            r=4,
            reset_on_terminate=True,
            dropout=0.0,  # No dropout for inference speed
        )

        # Initialize AGaLiTe memory
        self.agalite_memory = None

        # Output heads
        self.actor = init_layer(nn.Linear(input_size, sum(env.single_action_space.nvec)), std=0.01)
        self.critic = init_layer(nn.Linear(input_size, 1), std=1)

        # Register normalization buffer
        max_vec = torch.tensor(
            [9, 1, 1, 10, 3, 254, 1, 1, 235, 8, 9, 250, 29, 1, 1, 8, 1, 1, 6, 3, 1, 2],
            dtype=torch.float32,
        )[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Encode observations into features."""
        B = observations.shape[0]

        # Process token observations
        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")
            B = observations.shape[0]

        # Handle invalid tokens
        observations = observations.clone()
        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        x_coords = ((coords_byte >> 4) & 0x0F).long()
        y_coords = (coords_byte & 0x0F).long()
        atr_indices = observations[..., 1].long()
        atr_values = observations[..., 2].float()

        # Create box observations
        box_obs = torch.zeros(
            (B, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=observations.device,
        )

        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )

        batch_idx = torch.arange(B, device=observations.device).unsqueeze(-1).expand_as(atr_values)
        box_obs[batch_idx[valid_tokens], atr_indices[valid_tokens], x_coords[valid_tokens], y_coords[valid_tokens]] = (
            atr_values[valid_tokens]
        )

        # Normalize and encode
        features = box_obs / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.cnn_encoder(features)

        encoded = torch.cat([self_features, cnn_features], dim=1)

        # Pass through AGaLiTe core (without time dimension - LSTM wrapper handles that)
        # Initialize memory if needed
        if self.agalite_memory is None:
            self.agalite_memory = BatchedAGaLiTe.initialize_memory(
                batch_size=B,
                n_layers=2,
                n_heads=4,
                d_head=64,
                eta=4,
                r=4,
                device=encoded.device,
            )

        # Check if batch size changed
        memory_batch_size = next(iter(next(iter(self.agalite_memory.values())))).shape[0]
        if memory_batch_size != B:
            # Reinitialize memory for new batch size
            self.agalite_memory = BatchedAGaLiTe.initialize_memory(
                batch_size=B,
                n_layers=2,
                n_heads=4,
                d_head=64,
                eta=4,
                r=4,
                device=encoded.device,
            )

        # Add time dimension for AGaLiTe
        encoded = encoded.unsqueeze(0)  # (1, B, features)
        terminations = torch.zeros(1, B, device=encoded.device)

        # Forward through AGaLiTe
        agalite_out, new_memory = self.agalite_core(encoded, terminations, self.agalite_memory)

        # Detach memory to prevent gradient accumulation
        self.agalite_memory = {
            key: tuple(tensor.detach() for tensor in layer_memory) for key, layer_memory in new_memory.items()
        }

        # Remove time dimension
        return agalite_out.squeeze(0)

    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hidden states into action logits and value."""
        return self.actor(hidden), self.critic(hidden)

    def reset_memory(self):
        """Reset AGaLiTe memory."""
        self.agalite_memory = None
