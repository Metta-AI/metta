"""
Exponential Memory Traces component for Partially Observable RL.

Implementation based on "Partially Observable Reinforcement Learning with Memory Traces"
by Eberhard et al. (ICML 2025).

The key insight is using exponential decay to compress historical information more efficiently
than sliding window approaches.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase

logger = logging.getLogger(__name__)


class ExponentialMemoryTraces(LayerBase):
    """
    Exponential memory traces for POMDP environments.

    Maintains multiple traces with different decay rates (lambda values) to capture
    information at different temporal scales.
    """

    def __init__(self, trace_dim: int = 128, lambda_values: List[float] = None, **cfg):
        if lambda_values is None:
            lambda_values = [0.0, 0.985]  # 0.0 = current obs only, 0.985 = long memory

        super().__init__(**cfg)

        self.trace_dim = trace_dim
        self.lambda_values = lambda_values
        self.num_traces = len(lambda_values)

        # Store lambda values as tensor for efficient computation
        self.register_buffer("lambdas", torch.tensor(lambda_values, dtype=torch.float32))

    def _initialize(self):
        """Initialize the component after source components are available."""
        if self._sources is None or len(self._sources) == 0:
            raise ValueError("ExponentialMemoryTraces requires at least one source component")

        # Get input size from first source
        input_size = self._in_tensor_shapes[0][-1]  # Last dimension is feature size

        # Create input projection network
        self.input_projection = nn.Linear(input_size, self.trace_dim)

        # Network to combine traces into output
        self.trace_combiner = nn.Sequential(
            nn.Linear(self.num_traces * self.trace_dim, self.trace_dim),
            nn.ReLU(),
            nn.Linear(self.trace_dim, self.trace_dim),
        )

        # Set output shape
        self._out_tensor_shape = [self.trace_dim]

        logger.info(
            f"ExponentialMemoryTraces setup complete: {input_size} -> {self.trace_dim}, "
            f"{self.num_traces} traces with lambdas {self.lambda_values}"
        )

    def _forward(self, td: TensorDict):
        """Forward pass with exponential memory update."""
        # Get input from source component
        x = td[self._sources[0]["name"]]
        batch_size = x.shape[0]
        device = x.device

        # Project input to trace dimension
        phi = self.input_projection(x)  # (batch_size, trace_dim)

        # Get previous memory traces from tensordict if available
        memory_key = f"{self._name}_memory"
        if memory_key in td:
            memory_traces = td[memory_key]
        else:
            # Initialize traces if not provided
            memory_traces = torch.zeros(batch_size, self.num_traces, self.trace_dim, device=device, dtype=x.dtype)

        # Update traces using exponential decay formula: (1-λ)*φ + λ*old_trace
        # Broadcast computation for all traces simultaneously
        lambdas = self.lambdas.view(1, -1, 1)  # (1, num_traces, 1)
        phi_expanded = phi.unsqueeze(1).expand(-1, self.num_traces, -1)  # (batch_size, num_traces, trace_dim)

        updated_traces = (1 - lambdas) * phi_expanded + lambdas * memory_traces

        # Combine traces for output
        traces_flat = updated_traces.view(batch_size, -1)  # (batch_size, num_traces * trace_dim)
        output = self.trace_combiner(traces_flat)  # (batch_size, trace_dim)

        # Store outputs in tensordict
        td[self._name] = output
        td[memory_key] = updated_traces

        return td

    def compute_trace_statistics(self, memory_traces: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute statistics for monitoring trace behavior.

        Args:
            memory_traces: Memory traces of shape (batch_size, num_traces, trace_dim)

        Returns:
            Dictionary with trace statistics
        """
        with torch.no_grad():
            # Compute norms for each trace
            trace_norms = torch.norm(memory_traces, dim=-1)  # (batch_size, num_traces)

            # Compute similarities between traces
            similarities = []
            for i in range(self.num_traces):
                for j in range(i + 1, self.num_traces):
                    trace_i = memory_traces[:, i, :]  # (batch_size, trace_dim)
                    trace_j = memory_traces[:, j, :]  # (batch_size, trace_dim)

                    # Cosine similarity
                    sim = F.cosine_similarity(trace_i, trace_j, dim=-1)  # (batch_size,)
                    similarities.append(sim.mean().item())

            return {
                "trace_norms_mean": trace_norms.mean(dim=0),  # (num_traces,)
                "trace_norms_std": trace_norms.std(dim=0),  # (num_traces,)
                "trace_similarities": torch.tensor(similarities),
                "total_trace_norm": torch.norm(memory_traces.view(memory_traces.shape[0], -1), dim=-1).mean(),
            }


class MemoryTraceProcessor(LayerBase):
    """
    Helper component to process and integrate memory traces with the main network flow.

    This component sits between the memory traces and the core LSTM/network components.
    """

    def __init__(
        self,
        integration_mode: str = "concat",  # "concat", "add", "attention"
        **cfg,
    ):
        super().__init__(**cfg)
        self.integration_mode = integration_mode

    def _initialize(self):
        """Initialize the component after source components are available."""
        if self._sources is None or len(self._sources) != 2:
            raise ValueError("MemoryTraceProcessor requires exactly 2 sources: [base_features, memory_traces]")

        # Get sizes from source shapes
        base_size = self._in_tensor_shapes[0][-1]  # Last dimension is feature size
        memory_size = self._in_tensor_shapes[1][-1]

        if self.integration_mode == "concat":
            self._out_tensor_shape = [base_size + memory_size]
            self.integration_layer = nn.Identity()  # Just concatenate

        elif self.integration_mode == "add":
            if base_size != memory_size:
                # Project to same size
                self.integration_layer = nn.Linear(memory_size, base_size)
                self._out_tensor_shape = [base_size]
            else:
                self.integration_layer = nn.Identity()
                self._out_tensor_shape = [base_size]

        elif self.integration_mode == "attention":
            # Use attention to combine base features with memory
            self.integration_layer = nn.MultiheadAttention(embed_dim=base_size, num_heads=8, batch_first=True)
            # Project memory to match base dimension if needed
            if memory_size != base_size:
                self.memory_projection = nn.Linear(memory_size, base_size)
            self._out_tensor_shape = [base_size]
        else:
            raise ValueError(f"Unknown integration_mode: {self.integration_mode}")

        logger.info(
            f"MemoryTraceProcessor setup: {self.integration_mode} mode, "
            f"base_size={base_size}, memory_size={memory_size}, output_size={self._out_tensor_shape}"
        )

    def _forward(self, td: TensorDict):
        """Integrate base features with memory traces."""
        base_features = td[self._sources[0]["name"]]
        memory_traces = td[self._sources[1]["name"]]

        if self.integration_mode == "concat":
            output = torch.cat([base_features, memory_traces], dim=-1)

        elif self.integration_mode == "add":
            projected_memory = self.integration_layer(memory_traces)
            output = base_features + projected_memory

        elif self.integration_mode == "attention":
            # Project memory if needed
            if hasattr(self, "memory_projection"):
                memory_traces = self.memory_projection(memory_traces)

            # Add sequence dimension for attention (batch_size, 1, dim)
            base_seq = base_features.unsqueeze(1)
            memory_seq = memory_traces.unsqueeze(1)

            # Apply attention: use memory as key/value, base as query
            attended, _ = self.integration_layer(query=base_seq, key=memory_seq, value=memory_seq)

            # Remove sequence dimension and add residual connection
            output = base_features + attended.squeeze(1)
        else:
            raise RuntimeError(f"Unknown integration_mode: {self.integration_mode}")

        td[self._name] = output
        return td
