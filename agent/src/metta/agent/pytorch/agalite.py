import logging
from typing import Dict, Optional, Tuple

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.agalite_batched import BatchedAGaLiTe

logger = logging.getLogger(__name__)


class Agalite(pufferlib.models.LSTMWrapper):
    """Hybrid AGaLiTe-LSTM architecture for efficient RL training.
    
    This uses AGaLiTe's sophisticated attention-based observation encoding
    combined with LSTM for temporal processing. This hybrid approach provides:
    - AGaLiTe's powerful observation processing with linear attention
    - LSTM's efficient O(1) temporal state updates
    - Compatibility with PufferLib's training infrastructure
    
    Note: This is not the pure AGaLiTe transformer from the paper, but a
    practical hybrid that achieves better training efficiency."""

    def __init__(self, env, policy=None, input_size=256, hidden_size=256):
        if policy is None:
            policy = AgalitePolicy(env, input_size=input_size, hidden_size=hidden_size)
        
        # Use PufferLib's LSTM wrapper for consistency with working agents
        super().__init__(env, policy, input_size, hidden_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize action conversion tensors (will be set by MettaAgent)
        self.action_index_tensor = None
        self.cum_action_max_params = None
        
        # Move to device
        self.to(self.device)

    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass compatible with MettaAgent expectations."""
        observations = td["env_obs"].to(self.device)
        
        # Initialize LSTM state if needed
        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}
        
        # Prepare LSTM state
        lstm_h = state.get("lstm_h")
        lstm_c = state.get("lstm_c")
        
        # Handle batch dimensions for BPTT
        B = observations.shape[0]
        if observations.dim() == 4:  # (B, T, H, W)
            TT = observations.shape[1]
            observations = observations.reshape(B * TT, *observations.shape[2:])
        else:
            TT = 1
        
        # Check if we need to initialize or adjust LSTM state
        if lstm_h is not None and lstm_c is not None:
            # Ensure state tensors are on correct device and have correct shape
            lstm_h = lstm_h.to(self.device)
            lstm_c = lstm_c.to(self.device)
            
            # Add sequence dimension if needed (from stored state)
            if lstm_h.dim() == 2:  # (B, hidden_size)
                lstm_h = lstm_h.unsqueeze(0)  # -> (1, B, hidden_size)
                lstm_c = lstm_c.unsqueeze(0)  # -> (1, B, hidden_size)
            
            # Handle batch size mismatches (e.g., from BPTT)
            if lstm_h.shape[1] != B:
                # State batch size doesn't match - reinitialize
                lstm_h = None
                lstm_c = None
                lstm_state = None
            else:
                lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None
        
        # Encode observations through policy
        hidden = self.policy.encode_observations(observations, state)
        
        # LSTM forward pass
        if TT > 1:
            # Reshape for LSTM: (B*TT, hidden) -> (TT, B, hidden)
            hidden = hidden.view(B, TT, -1).transpose(0, 1)
        else:
            # Add time dimension: (B, hidden) -> (1, B, hidden)
            hidden = hidden.unsqueeze(0)
        
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
        
        # Flatten back for decoding
        if TT > 1:
            flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)
        else:
            flat_hidden = lstm_output.squeeze(0)
        
        # Decode actions and value
        logits, value = self.policy.decode_actions(flat_hidden)
        
        # Handle action sampling/evaluation
        if action is None:
            # Inference mode - sample actions
            action_log_probs = F.log_softmax(logits, dim=-1)
            action_probs = torch.exp(action_log_probs)
            
            sampled_actions = torch.multinomial(action_probs, num_samples=1).view(-1)
            batch_indices = torch.arange(sampled_actions.shape[0], device=sampled_actions.device)
            selected_log_probs = action_log_probs[batch_indices, sampled_actions]
            
            # Convert to action pairs
            converted_actions = self._convert_logit_index_to_action(sampled_actions)
            
            td["actions"] = converted_actions.to(dtype=torch.int32)
            td["act_log_prob"] = selected_log_probs
            td["values"] = value.flatten()
            td["full_log_probs"] = action_log_probs
            
            # Store LSTM state (squeeze the sequence dimension for inference)
            td["lstm_h"] = new_lstm_h.squeeze(0).detach()  # Remove sequence dimension
            td["lstm_c"] = new_lstm_c.squeeze(0).detach()  # Remove sequence dimension
            
        else:
            # Training mode - evaluate given actions
            action = action.to(self.device)
            if action.dim() == 3:  # (B, T, A)
                action = action.view(B * TT, -1)
            
            # Convert actions to logit indices
            action_logit_index = self._convert_action_to_logit_index(action)
            
            # Compute log probs and entropy
            action_log_probs = F.log_softmax(logits, dim=-1)
            action_probs = torch.exp(action_log_probs)
            
            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            selected_log_probs = action_log_probs[batch_indices, action_logit_index]
            
            entropy = -(action_probs * action_log_probs).sum(dim=-1)
            
            # Reshape for BPTT if needed
            if TT > 1:
                td["act_log_prob"] = selected_log_probs.view(B, TT)
                td["entropy"] = entropy.view(B, TT)
                td["value"] = value.view(B, TT)
                td["full_log_probs"] = action_log_probs.view(B, TT, -1)
            else:
                td["act_log_prob"] = selected_log_probs
                td["entropy"] = entropy
                td["value"] = value.flatten()
                td["full_log_probs"] = action_log_probs
        
        return td

    def clip_weights(self):
        """Clip weights to prevent gradient explosion."""
        for p in self.parameters():
            p.data.clamp_(-1, 1)

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        if self.action_index_tensor is None:
            raise RuntimeError("action_index_tensor not initialized. Call activate_actions first.")
        return self.action_index_tensor[action_logit_index]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        if self.cum_action_max_params is None:
            raise RuntimeError("cum_action_max_params not initialized. Call activate_actions first.")
        
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return cumulative_sum + action_params


class AgalitePolicy(nn.Module):
    """Inner policy using AGaLiTe architecture."""
    
    def __init__(self, env, input_size=256, hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = False  # Required by PufferLib
        self.action_space = env.single_action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Observation parameters
        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22
        
        # Create observation encoders
        self.cnn_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.num_layers, 128, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(128, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(128, input_size // 2)),
            nn.ReLU(),
        )
        
        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, input_size // 2)),
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
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(input_size, sum(env.single_action_space.nvec)), std=0.01
        )
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(input_size, 1), std=1
        )
        
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
        box_obs[batch_idx[valid_tokens], atr_indices[valid_tokens], 
                x_coords[valid_tokens], y_coords[valid_tokens]] = atr_values[valid_tokens]
        
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
            key: tuple(tensor.detach() for tensor in layer_memory)
            for key, layer_memory in new_memory.items()
        }
        
        # Remove time dimension
        return agalite_out.squeeze(0)

    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hidden states into action logits and value."""
        return self.actor(hidden), self.critic(hidden)

    def reset_memory(self):
        """Reset AGaLiTe memory."""
        self.agalite_memory = None