"""
Faithful AGaLiTe implementation for Metta that properly integrates with the training infrastructure.
This version uses the TransformerWrapper for proper BPTT handling.
"""

import logging
from typing import Dict, Optional, Tuple

import einops
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from pufferlib.pytorch import layer_init as init_layer

from metta.agent.modules.agalite_layers import AttentionAGaLiTeLayer, RecurrentLinearTransformerEncoder
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class AGaLiTeCore(nn.Module):
    """
    Full AGaLiTe transformer model with proper memory handling.
    Processes entire BPTT sequences as context.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_head = d_head
        self.d_ffc = d_ffc
        self.n_heads = n_heads
        self.eta = eta
        self.r = r

        self.encoders = nn.ModuleList()
        for layer in range(n_layers):
            use_dense = layer == 0  # Use dense layer for first layer
            encoder = RecurrentLinearTransformerEncoder(
                d_model=d_model,
                d_head=d_head,
                d_ffc=d_ffc,
                n_heads=n_heads,
                eta=eta,
                r=r,
                use_dense=use_dense,
                reset_hidden_on_terminate=reset_on_terminate,
                dropout=dropout,
            )
            self.encoders.append(encoder)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, memory: Dict[str, Tuple]
    ) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
        """Forward pass for AGaLiTe."""
        u_i = inputs
        new_memory = {}

        for layer_idx, encoder in enumerate(self.encoders):
            layer_key = f"layer_{layer_idx + 1}"
            u_i, memory_updated = encoder(u_i, terminations, memory[layer_key])
            new_memory[layer_key] = memory_updated

        return u_i, new_memory

    @staticmethod
    def initialize_memory(
        batch_size: int, n_layers: int, n_heads: int, d_head: int, eta: int, r: int, device: torch.device = None
    ) -> Dict[str, Tuple]:
        """Initialize memory for all layers."""
        memory_dict = {}
        for layer in range(1, n_layers + 1):
            memory_dict[f"layer_{layer}"] = AttentionAGaLiTeLayer.initialize_memory(
                batch_size, n_heads, d_head, eta, r, device
            )
        return memory_dict


class AGaLiTePolicy(nn.Module):
    """
    AGaLiTe policy network that implements encode_observations and decode_actions.
    This is designed to work with the TransformerWrapper.
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 4,
        eta: int = 4,
        r: int = 8,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_space = env.single_action_space

        # AGaLiTe parameters
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.eta = eta
        self.r = r
        self.reset_on_terminate = reset_on_terminate

        # Required by TransformerWrapper
        self.is_continuous = False
        self.hidden_size = d_model

        # Observation encoding (same as Fast agent for token observations)
        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        # Token to grid conversion
        self.cnn1 = init_layer(nn.Conv2d(22, 64, kernel_size=5, stride=3))
        self.cnn2 = init_layer(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        test_input = torch.zeros(1, 22, 11, 11)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()
        self.fc1 = init_layer(nn.Linear(self.flattened_size, 128))
        self.encoded_obs = init_layer(nn.Linear(128, d_model))

        # Create the AGaLiTe transformer
        self.transformer = AGaLiTeCore(
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
        self.critic_1 = init_layer(nn.Linear(d_model, 1024))
        self.value_head = init_layer(nn.Linear(1024, 1), std=1.0)
        self.actor_1 = init_layer(nn.Linear(d_model, 512))
        self.action_embeddings = nn.Embedding(100, 16)

        # Action heads
        if hasattr(self.action_space, "nvec"):
            action_nvec = self.action_space.nvec
        else:
            action_nvec = [100]

        self.actor_heads = nn.ModuleList([init_layer(nn.Linear(512 + 16, n), std=0.01) for n in action_nvec])

        # Normalization buffer
        max_vec = torch.tensor(
            [
                9.0,
                1.0,
                1.0,
                10.0,
                3.0,
                254.0,
                1.0,
                1.0,
                235.0,
                8.0,
                9.0,
                250.0,
                29.0,
                1.0,
                1.0,
                8.0,
                1.0,
                1.0,
                6.0,
                3.0,
                1.0,
                2.0,
            ],
            dtype=torch.float32,
        )[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

    def network_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CNN feature extraction from grid observations."""
        x = x / self.max_vec
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.encoded_obs(x)
        return x

    def encode_observations(self, observations: torch.Tensor, state: Optional[Dict] = None) -> torch.Tensor:
        """Encode token observations to hidden representation."""
        # Convert from Byte to Float and move to device
        observations = observations.float()
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]

        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"
        token_observations[token_observations == 255] = 0

        # Convert tokens to grid representation
        coords_byte = token_observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = token_observations[..., 1].long()
        atr_values = token_observations[..., 2].float()

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )
        batch_indices = torch.arange(B * TT, device=token_observations.device).unsqueeze(-1).expand_as(atr_values)

        valid_tokens = coords_byte != 0xFF
        valid_tokens = valid_tokens & (x_coord_indices < self.out_width) & (y_coord_indices < self.out_height)
        valid_tokens = valid_tokens & (atr_indices < self.num_layers)

        box_obs[
            batch_indices[valid_tokens],
            atr_indices[valid_tokens],
            x_coord_indices[valid_tokens],
            y_coord_indices[valid_tokens],
        ] = atr_values[valid_tokens]

        # Encode with CNN
        hidden = self.network_forward(box_obs)

        return hidden

    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hidden representation to action logits and value."""
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        actor_features = self.actor_1(hidden)

        action_embed = self.action_embeddings.weight.mean(dim=0).unsqueeze(0).expand(actor_features.shape[0], -1)
        combined_features = torch.cat([actor_features, action_embed], dim=-1)

        logits = torch.cat([head(combined_features) for head in self.actor_heads], dim=-1)

        return logits, value

    def initialize_memory(self, batch_size: int) -> Dict:
        """Initialize AGaLiTe memory for a batch."""
        return AGaLiTeCore.initialize_memory(
            batch_size=batch_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_head=self.d_head,
            eta=self.eta,
            r=self.r,
        )


class AGaLiTe(PyTorchAgentMixin, TransformerWrapper):
    """
    AGaLiTe implementation using TransformerWrapper for proper BPTT handling.

    This provides:
    - Proper memory state management across BPTT segments
    - Correct batch dimension handling for vectorized environments
    - Termination-aware state resets
    - Full compatibility with Metta's training infrastructure
    - Weight management and action conversion via PyTorchAgentMixin
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 4,
        eta: int = 4,
        r: int = 8,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        """Initialize AGaLiTe with mixin support.
        
        Args:
            env: Environment
            d_model: Model dimension
            d_head: Head dimension
            d_ffc: Feedforward dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            eta: AGaLiTe eta parameter
            r: AGaLiTe r parameter
            reset_on_terminate: Whether to reset memory on termination
            dropout: Dropout rate
            **kwargs: Configuration parameters handled by mixin (clip_range, analyze_weights_interval, etc.)
        """
        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)
        
        # Create the AGaLiTe policy
        policy = AGaLiTePolicy(
            env=env,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            n_layers=n_layers,
            eta=eta,
            r=r,
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
        )

        # Initialize with TransformerWrapper
        super().__init__(env, policy, hidden_size=d_model)
        
        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)

    def forward(self, td: TensorDict, state: Optional[Dict] = None, action: Optional[torch.Tensor] = None):
        """Forward pass compatible with MettaAgent expectations."""
        observations = td["env_obs"]

        # Initialize state if needed (handle both None and lazy init cases)
        if state is None or state.get("needs_init", False):
            B = observations.shape[0]
            state = self.reset_memory(B, observations.device)

        # Store terminations if available
        if "dones" in td:
            state["terminations"] = td["dones"]
            
        # Set critical TensorDict fields using mixin
        # Note: For transformer, we still set these but handle differently than LSTM
        B, TT = self.set_tensordict_fields(td, observations)

        # Determine if we're in training or inference mode
        if action is None:
            # Inference mode
            logits, values = self.forward_eval(observations, state)
            
            # Use mixin for inference mode processing
            td = self.forward_inference(td, logits, values)

        else:
            # Training mode - use parent's forward for BPTT
            logits, values = super().forward(observations, state)

            # Use mixin for training mode processing
            # Note: TransformerWrapper handles the reshaping of values
            td = self.forward_training(td, action, logits, values)
            
            # Override value key for transformer (TransformerWrapper uses "value" not "values")
            td["value"] = values

        return td
