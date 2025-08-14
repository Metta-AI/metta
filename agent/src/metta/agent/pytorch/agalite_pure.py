"""
Pure AGaLiTe (Approximate Gated Linear Transformer) implementation for Metta.
This is a faithful reproduction of the paper's architecture without LSTM dependencies.
"""

import logging
from typing import Dict, Optional, Tuple

import einops
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.modules.agalite_batched import BatchedAGaLiTe

logger = logging.getLogger(__name__)


class AgalitePure(nn.Module):
    """Pure AGaLiTe transformer-based recurrent policy.
    
    This implements the AGaLiTe architecture from the paper:
    - Uses linear attention with gating for efficient recurrence
    - No LSTM dependency - pure transformer-based
    - Handles vectorized environments and BPTT properly
    """

    def __init__(
        self,
        env,
        n_layers: int = 4,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        eta: int = 4,
        r: int = 8,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = env.single_action_space
        self.obs_shape = env.single_observation_space.shape
        
        # AGaLiTe parameters
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.eta = eta
        self.r = r
        
        # Required by PufferLib
        self.is_continuous = False
        self.hidden_size = d_model  # For compatibility
        
        # Observation encoding parameters
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
            pufferlib.pytorch.layer_init(nn.Linear(128, d_model // 2)),
            nn.ReLU(),
        )
        
        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, d_model // 2)),
            nn.ReLU(),
        )
        
        # Create the AGaLiTe transformer
        self.agalite = BatchedAGaLiTe(
            n_layers=n_layers,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            eta=eta,
            r=r,
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
        )
        
        # Output heads
        if hasattr(self.action_space, 'nvec'):
            # Multi-discrete action space
            self.actor = nn.ModuleList([
                pufferlib.pytorch.layer_init(nn.Linear(d_model, n), std=0.01) 
                for n in self.action_space.nvec
            ])
        else:
            # Single discrete action space
            num_actions = self.action_space.n if hasattr(self.action_space, 'n') else sum(self.action_space.nvec)
            self.actor = pufferlib.pytorch.layer_init(nn.Linear(d_model, num_actions), std=0.01)
        
        self.critic = pufferlib.pytorch.layer_init(nn.Linear(d_model, 1), std=1)
        
        # Register normalization buffer
        max_vec = torch.tensor(
            [9, 1, 1, 10, 3, 254, 1, 1, 235, 8, 9, 250, 29, 1, 1, 8, 1, 1, 6, 3, 1, 2],
            dtype=torch.float32,
        )[None, :, None, None]
        self.register_buffer("max_vec", max_vec)
        
        # Action conversion tensors (will be set by MettaAgent)
        self.action_index_tensor = None
        self.cum_action_max_params = None
        
        self.to(self.device)
    
    def forward_eval(self, observations: torch.Tensor, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for inference (single timestep).
        
        Args:
            observations: Observations of shape (B, ...)
            state: Dictionary containing AGaLiTe memory state
            
        Returns:
            logits: Action logits
            values: Value estimates
        """
        B = observations.shape[0]
        observations = observations.to(self.device)
        
        # Encode observations
        hidden = self.encode_observations(observations)
        
        # Get or initialize AGaLiTe memory
        memory = self._get_or_init_memory(state, B)
        
        # Add time dimension for AGaLiTe (expects T, B, D)
        hidden = hidden.unsqueeze(0)  # (1, B, d_model)
        terminations = torch.zeros(1, B, device=self.device)
        
        # Forward through AGaLiTe
        agalite_out, new_memory = self.agalite(hidden, terminations, memory)
        
        # Remove time dimension
        hidden = agalite_out.squeeze(0)  # (B, d_model)
        
        # Decode actions and value
        logits, values = self.decode_actions(hidden)
        
        # Store updated memory (detached to prevent gradient accumulation)
        self._store_memory(state, new_memory)
        
        return logits, values
    
    def forward_train(self, observations: torch.Tensor, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training (handles sequences with BPTT).
        
        Args:
            observations: Observations of shape (B, T, ...) or (B, ...)
            state: Dictionary containing AGaLiTe memory state
            
        Returns:
            logits: Action logits
            values: Value estimates of shape (B, T)
        """
        x = observations.to(self.device)
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)
        
        # Determine batch and time dimensions
        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)
        
        # Reshape for encoding: (B*T, ...)
        x = x.reshape(B * TT, *space_shape)
        
        # Encode observations
        hidden = self.encode_observations(x)  # (B*T, d_model)
        
        # Get or initialize memory
        memory = self._get_or_init_memory(state, B)
        
        # Reshape for AGaLiTe: (B*T, d_model) -> (T, B, d_model)
        if TT > 1:
            hidden = hidden.view(B, TT, -1).transpose(0, 1)
            # Get terminations if available
            terminations = state.get('terminations', torch.zeros(TT, B, device=self.device))
        else:
            hidden = hidden.unsqueeze(0)  # (1, B, d_model)
            terminations = torch.zeros(1, B, device=self.device)
        
        # Forward through AGaLiTe
        agalite_out, new_memory = self.agalite(hidden, terminations, memory)
        
        # Reshape back: (T, B, d_model) -> (B*T, d_model)
        if TT > 1:
            flat_hidden = agalite_out.transpose(0, 1).reshape(B * TT, -1)
        else:
            flat_hidden = agalite_out.squeeze(0)
        
        # Decode actions and value
        logits, values = self.decode_actions(flat_hidden)
        
        # Reshape values for output
        values = values.reshape(B, TT)
        
        # Store updated memory (detached)
        self._store_memory(state, new_memory)
        
        return logits, values
    
    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations into features."""
        B = observations.shape[0]
        
        # Handle token observations
        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")
            B = observations.shape[0]
        
        # Process observations
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
            device=self.device,
        )
        
        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )
        
        batch_idx = torch.arange(B, device=self.device).unsqueeze(-1).expand_as(atr_values)
        box_obs[
            batch_idx[valid_tokens],
            atr_indices[valid_tokens],
            x_coords[valid_tokens],
            y_coords[valid_tokens]
        ] = atr_values[valid_tokens]
        
        # Normalize and encode
        features = box_obs / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.cnn_encoder(features)
        
        return torch.cat([self_features, cnn_features], dim=1)
    
    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hidden states into action logits and value."""
        if isinstance(self.actor, nn.ModuleList):
            logits = torch.cat([head(hidden) for head in self.actor], dim=-1)
        else:
            logits = self.actor(hidden)
        
        values = self.critic(hidden)
        return logits, values
    
    def _get_or_init_memory(self, state: Dict, batch_size: int) -> Dict:
        """Get AGaLiTe memory from state or initialize new."""
        if 'agalite_memory' in state and state['agalite_memory'] is not None:
            return state['agalite_memory']
        
        # Initialize new memory
        memory = BatchedAGaLiTe.initialize_memory(
            batch_size=batch_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_head=self.d_head,
            eta=self.eta,
            r=self.r,
            device=self.device,
        )
        return memory
    
    def _store_memory(self, state: Dict, memory: Dict):
        """Store AGaLiTe memory in state (detached to prevent gradient accumulation)."""
        detached_memory = {}
        for key, layer_memory in memory.items():
            detached_memory[key] = tuple(tensor.detach() for tensor in layer_memory)
        state['agalite_memory'] = detached_memory
    
    # MettaAgent compatibility methods
    def forward(self, td: TensorDict, state=None, action=None) -> TensorDict:
        """Forward pass compatible with MettaAgent."""
        # Handle BPTT reshaping first
        original_shape = td.batch_size
        if td.batch_dims > 1:
            B, TT = td.batch_size
            total_batch = B * TT
            td = td.reshape(total_batch)
            td.set("bptt", torch.full((total_batch,), TT, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((total_batch,), B, device=td.device, dtype=torch.long))
        else:
            B = td.batch_size[0] if hasattr(td.batch_size, "__getitem__") else td.batch_size
            TT = 1
            total_batch = B
            td.set("bptt", torch.full((B,), 1, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=td.device, dtype=torch.long))
        
        observations = td["env_obs"].to(self.device)
        
        if state is None:
            state = {}
        
        # Forward based on whether we're in training or inference
        if TT == 1:  # Single timestep - inference
            logits, values = self.forward_eval(observations, state)
        else:  # Multiple timesteps - training with BPTT
            # Reshape observations for BPTT
            observations = observations.view(B, TT, *observations.shape[1:])
            logits, values = self.forward_train(observations, state)
        
        # Handle action sampling/evaluation
        if action is None:
            # Inference mode
            action_log_probs = F.log_softmax(logits, dim=-1)
            action_probs = torch.exp(action_log_probs)
            
            sampled_actions = torch.multinomial(action_probs, num_samples=1).view(-1)
            batch_indices = torch.arange(sampled_actions.shape[0], device=sampled_actions.device)
            selected_log_probs = action_log_probs[batch_indices, sampled_actions]
            
            # Convert to action pairs if needed
            if self.action_index_tensor is not None:
                converted_actions = self.action_index_tensor[sampled_actions]
            else:
                converted_actions = sampled_actions.unsqueeze(-1)
            
            td["actions"] = converted_actions.to(dtype=torch.int32)
            td["act_log_prob"] = selected_log_probs
            td["values"] = values.flatten()
            td["full_log_probs"] = action_log_probs
        else:
            # Training mode
            action = action.to(self.device)
            B = observations.shape[0]
            TT = observations.shape[1] if observations.dim() > 3 else 1
            
            if action.dim() == 3:  # (B, T, A)
                action = action.view(B * TT, -1)
            
            # Convert actions to logit indices if needed
            if self.cum_action_max_params is not None:
                action_type_numbers = action[:, 0].long()
                action_params = action[:, 1].long()
                action_logit_index = self.cum_action_max_params[action_type_numbers] + action_params
            else:
                action_logit_index = action.squeeze(-1).long()
            
            # Compute log probs and entropy
            action_log_probs = F.log_softmax(logits, dim=-1)
            action_probs = torch.exp(action_log_probs)
            
            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            selected_log_probs = action_log_probs[batch_indices, action_logit_index]
            
            entropy = -(action_probs * action_log_probs).sum(dim=-1)
            
            # Store in TensorDict
            if TT > 1:
                td["act_log_prob"] = selected_log_probs.view(B, TT)
                td["entropy"] = entropy.view(B, TT)
                td["value"] = values
                td["full_log_probs"] = action_log_probs.view(B, TT, -1)
            else:
                td["act_log_prob"] = selected_log_probs
                td["entropy"] = entropy
                td["value"] = values.flatten()
                td["full_log_probs"] = action_log_probs
        
        # Store AGaLiTe memory in td (only if it exists)
        # Note: Memory storage is handled differently in MettaAgent
        # We don't need to serialize here as state is managed externally
        
        # Reshape back to original if we modified it
        if original_shape != td.batch_size:
            td = td.reshape(original_shape)
        
        return td
    
    def _serialize_memory(self, memory_dict: Dict) -> torch.Tensor:
        """Serialize memory dictionary into a single tensor for storage."""
        tensors = []
        for layer_idx in range(1, self.n_layers + 1):
            layer_memory = memory_dict[f"layer_{layer_idx}"]
            for tensor in layer_memory:
                batch_size = tensor.shape[0]
                flattened = tensor.view(batch_size, -1)
                tensors.append(flattened)
        return torch.cat(tensors, dim=1)
    
    def clip_weights(self):
        """Clip weights to prevent gradient explosion."""
        for p in self.parameters():
            p.data.clamp_(-1, 1)