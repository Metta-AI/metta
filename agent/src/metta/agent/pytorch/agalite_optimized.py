"""
AGaLiTe Optimized - Balanced performance variant with clean implementation.
Uses FastAGaLiTeLayer with enhanced parameters for better metrics.
"""

import logging
from typing import Dict

import torch
from torch import nn

from metta.agent.modules.agalite_fast import FastAGaLiTeLayer
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.agalite import AGaLiTePolicy
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin
from tensordict import TensorDict
from typing import Optional

logger = logging.getLogger(__name__)


class AGaLiTeOptimized(PyTorchAgentMixin, TransformerWrapper):
    """
    Optimized AGaLiTe - Balanced between speed and performance.

    Uses FastAGaLiTeLayer architecture with:
    - eta=3, r=6 for better capacity than fast mode
    - 2 layers for reasonable depth
    - Small dropout for generalization

    Expected performance: ~100k SPS with better metrics than fast mode
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        eta: int = 3,  # Higher than fast (2) but lower than full (4)
        r: int = 6,  # Higher than fast (4) but lower than full (8)
        dropout: float = 0.05,
        **kwargs,
    ):
        """Initialize optimized AGaLiTe with balanced parameters."""
        logger.info(f"Creating AGaLiTeOptimized with eta={eta}, r={r}, layers={n_layers}")

        # Extract mixin parameters
        mixin_params = self.extract_mixin_params(kwargs)

        # Create policy with custom transformer
        policy = OptimizedPolicy(
            env=env,
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            n_layers=n_layers,
            eta=eta,
            r=r,
            dropout=dropout,
        )

        # Initialize with TransformerWrapper
        super().__init__(env, policy, hidden_size=d_model)

        # Initialize mixin
        self.init_mixin(**mixin_params)

    @torch._dynamo.disable  # Avoid graph breaks with recurrent state
    def forward(self, td: TensorDict, state: Optional[Dict] = None, action: Optional[torch.Tensor] = None):
        """Forward pass with proper TensorDict handling.

        Follows the same pattern as AGaLiTe base implementation.
        """
        observations = td["env_obs"]

        # Determine dimensions from observations
        if observations.dim() == 4:  # Training
            B = observations.shape[0]
            TT = observations.shape[1]
        else:  # Inference
            B = observations.shape[0]
            TT = 1

        # Initialize state if needed
        if state is None or state.get("needs_init", False):
            state = self.reset_memory(B, observations.device)

        # Store terminations if available
        if "dones" in td:
            state["terminations"] = td["dones"]

        # Reshape TD for training if needed
        if observations.dim() == 4 and td.batch_dims > 1:
            td = td.reshape(B * TT)

        # Set critical TensorDict fields using mixin
        self.set_tensordict_fields(td, observations)

        # Determine if we're in training or inference mode
        if action is None:
            # Inference mode
            logits, values = self.forward_eval(observations, state)
            td = self.forward_inference(td, logits, values)
        else:
            # Training mode - use parent's forward for BPTT
            logits, values = super().forward(observations, state)

            # The mixin expects values to be flattened for training
            if values.dim() == 2:  # (B, T) from TransformerWrapper
                values_flat = values.flatten()
            else:
                values_flat = values

            td = self.forward_training(td, action, logits, values_flat)

        return td


class OptimizedPolicy(AGaLiTePolicy):
    """Optimized policy using FastAGaLiTeLayer directly."""

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        eta: int = 3,
        r: int = 6,
        dropout: float = 0.05,
        **kwargs,
    ):
        # Initialize base policy (skip transformer creation)
        nn.Module.__init__(self)

        # Copy necessary attributes from AGaLiTePolicy
        self.action_space = env.single_action_space
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.eta = eta
        self.r = r
        self.is_continuous = False
        self.hidden_size = d_model

        # Use AGaLiTePolicy's observation encoding
        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        # Reuse AGaLiTePolicy's CNN layers
        import numpy as np
        from pufferlib.pytorch import layer_init as init_layer

        self.cnn1 = init_layer(nn.Conv2d(22, 64, kernel_size=5, stride=3))
        self.cnn2 = init_layer(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        # Calculate flattened size
        test_input = torch.zeros(1, 22, 11, 11)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()
        self.fc1 = init_layer(nn.Linear(self.flattened_size, 128))
        self.encoded_obs = init_layer(nn.Linear(128, d_model))

        # Create optimized transformer
        self.transformer = OptimizedCore(
            n_layers=n_layers,
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            eta=eta,
            r=r,
            dropout=dropout,
        )

        # Output heads
        self.critic_1 = init_layer(nn.Linear(d_model, 1024), std=np.sqrt(2))
        self.value_head = init_layer(nn.Linear(1024, 1), std=1.0)
        self.actor_1 = init_layer(nn.Linear(d_model, 512), std=1.0)
        self.action_embeddings = nn.Embedding(100, 16)

        # Initialize action embeddings
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(self.action_embeddings.weight))
            self.action_embeddings.weight.mul_(0.1 / max_abs_value)

        # Action heads
        if hasattr(self.action_space, "nvec"):
            action_nvec = self.action_space.nvec
        else:
            action_nvec = [100]

        self.actor_heads = nn.ModuleList([init_layer(nn.Linear(512 + 16, n), std=0.01) for n in action_nvec])

        # Copy normalization buffer from AGaLiTePolicy
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

        # Copy methods from AGaLiTePolicy
        self.network_forward = AGaLiTePolicy.network_forward.__get__(self, OptimizedPolicy)
        self.encode_observations = AGaLiTePolicy.encode_observations.__get__(self, OptimizedPolicy)
        self.decode_actions = AGaLiTePolicy.decode_actions.__get__(self, OptimizedPolicy)

    def initialize_memory(self, batch_size: int) -> Dict:
        """Initialize memory with custom parameters."""
        device = next(self.parameters()).device
        return self.transformer.initialize_memory(batch_size, device)


class OptimizedCore(nn.Module):
    """Optimized transformer core using FastAGaLiTeLayer."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        n_heads: int,
        eta: int,
        r: int,
        dropout: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.eta = eta
        self.r = r

        # Create FastAGaLiTe layers
        self.layers = nn.ModuleList(
            [
                FastAGaLiTeLayer(
                    d_model=d_model,
                    head_num=n_heads,
                    head_dim=d_head,
                    eta=eta,
                    r=r,
                    reset_hidden_on_terminate=True,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs, terminations, memory):
        """Forward pass through layers with residual connections."""
        u_i = inputs
        new_memory = {}

        for i, layer in enumerate(self.layers):
            layer_key = f"layer_{i + 1}"
            residual = u_i
            attn_out, layer_memory = layer(u_i, terminations, memory[layer_key])
            u_i = residual + attn_out  # Residual connection
            new_memory[layer_key] = layer_memory

        return u_i, new_memory

    def initialize_memory(self, batch_size: int, device=None):
        """Initialize memory for all layers."""
        return {
            f"layer_{i + 1}": FastAGaLiTeLayer.initialize_memory(
                batch_size, self.n_heads, self.d_head, self.eta, self.r, device
            )
            for i in range(self.n_layers)
        }
