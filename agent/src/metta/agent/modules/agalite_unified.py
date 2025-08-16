"""
Unified AGaLiTe implementation that combines performance optimizations with full features.
This replaces the separate standard/fast modes with a single optimized implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from metta.agent.modules.agalite_optimized import discounted_sum
from metta.agent.modules.gru_gating import SimpleGRUGatingUnit


class UnifiedAGaLiTeLayer(nn.Module):
    """
    Unified AGaLiTe layer with all features and optimizations.
    Combines the efficiency of fast mode with the full feature set.
    """
    
    def __init__(
        self,
        d_model: int,
        head_num: int,
        head_dim: int,
        d_ffc: int,
        eta: int = 4,
        r: int = 8,
        reset_hidden_on_terminate: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
        use_layer_norm: bool = True,
        use_gru_gating: bool = True,
        use_ffc: bool = True,
        optimize_large_batch: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = head_dim
        self.d_ffc = d_ffc
        self.eta = eta
        self.r = r
        self.reset_hidden_on_terminate = reset_hidden_on_terminate
        self.eps = eps
        self.use_layer_norm = use_layer_norm
        self.use_gru_gating = use_gru_gating
        self.use_ffc = use_ffc
        self.optimize_large_batch = optimize_large_batch
        
        # Fused projection for efficiency
        total_proj_dim = head_num * head_dim * 5 + head_num * eta * 3
        self.fused_projection = nn.Linear(d_model, total_proj_dim)
        
        # Attention output projection
        self.attn_project = nn.Linear(head_num * head_dim, d_model)
        
        # Optional components
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        
        if use_gru_gating:
            self.gru1 = SimpleGRUGatingUnit(d_model, bias=2.0)
            if use_ffc:
                self.gru2 = SimpleGRUGatingUnit(d_model, bias=2.0)
        
        if use_ffc:
            self.ffc = nn.Sequential(
                nn.Linear(d_model, d_ffc),
                nn.ReLU(),
                nn.Linear(d_ffc, d_model),
                nn.ReLU() if use_gru_gating else nn.Identity()
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Pre-compute oscillatory frequencies
        self.register_buffer('omegas', torch.linspace(-math.pi, math.pi, r))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.orthogonal_(self.fused_projection.weight, gain=math.sqrt(2))
        nn.init.constant_(self.fused_projection.bias, 0)
        nn.init.orthogonal_(self.attn_project.weight, gain=1.0)
        nn.init.constant_(self.attn_project.bias, 0)
        
        if self.use_ffc:
            for layer in self.ffc:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                    nn.init.constant_(layer.bias, 0)
    
    def forward(
        self,
        inputs: torch.Tensor,
        terminations: torch.Tensor,
        memory: Tuple,
        residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass with full features and optimizations.
        
        Args:
            inputs: (T, B, d_model) input tensor
            terminations: (T, B) termination signals
            memory: Previous memory state
            residual: Optional residual connection from previous layer
        
        Returns:
            output: (T, B, d_model) output tensor
            new_memory: Updated memory state
        """
        T, B, _ = inputs.shape
        device = inputs.device
        
        # Store original input for residual connections
        inputs_orig = inputs
        if residual is not None:
            inputs = inputs + residual
        
        # Layer norm before attention (pre-norm)
        if self.use_layer_norm:
            inputs_normed = self.ln1(inputs)
        else:
            inputs_normed = inputs
        
        # Compute attention
        attn_out, new_memory = self._compute_attention(
            inputs_normed, terminations, memory
        )
        
        # Apply dropout
        attn_out = self.dropout(attn_out)
        
        # First GRU gating (attention output with input)
        if self.use_gru_gating:
            gating1_out = self.gru1(inputs, F.relu(attn_out))
        else:
            gating1_out = inputs + attn_out
        
        # Feed-forward component
        if self.use_ffc:
            # Layer norm before FFC
            if self.use_layer_norm:
                ffc_input = self.ln2(gating1_out)
            else:
                ffc_input = gating1_out
            
            ffc_out = self.ffc(ffc_input)
            ffc_out = self.dropout(ffc_out)
            
            # Second GRU gating (FFC output with previous)
            if self.use_gru_gating:
                output = self.gru2(gating1_out, ffc_out)
            else:
                output = gating1_out + ffc_out
        else:
            output = gating1_out
        
        return output, new_memory
    
    def _compute_attention(
        self,
        inputs: torch.Tensor,
        terminations: torch.Tensor,
        memory: Tuple
    ) -> Tuple[torch.Tensor, Tuple]:
        """Compute AGaLiTe attention with optimizations."""
        T, B, _ = inputs.shape
        device = inputs.device
        
        # Unpack memory
        tilde_k_prev, tilde_v_prev, s_prev, tick = memory
        
        # Single fused projection
        all_proj = self.fused_projection(inputs)
        
        # Split projections
        kqv_dim = self.head_num * self.head_dim * 5
        kqvbetagammas = all_proj[..., :kqv_dim]
        p1p2p3 = all_proj[..., kqv_dim:]
        
        # Reshape efficiently
        kqvbetagammas = kqvbetagammas.view(T, B, self.head_num, 5, self.head_dim)
        p1p2p3 = p1p2p3.view(T, B, self.head_num, 3, self.eta)
        
        # Extract components
        keys = kqvbetagammas[..., 0, :]
        queries = kqvbetagammas[..., 1, :]
        values = kqvbetagammas[..., 2, :]
        beta = torch.sigmoid(kqvbetagammas[..., 3, :])
        gammas = kqvbetagammas[..., 4, :]
        
        p1 = p1p2p3[..., 0, :]
        p2 = p1p2p3[..., 1, :]
        p3 = p1p2p3[..., 2, :]
        
        # Compute expanded features using batched operations
        keys_expanded = self._compute_outer_product(
            F.relu(keys), F.relu(p1), T, B
        )
        queries_expanded = self._compute_outer_product(
            F.relu(queries), F.relu(p2), T, B
        )
        gammas_expanded = self._compute_outer_product(
            torch.sigmoid(gammas), torch.sigmoid(p3), T, B
        )
        
        # Oscillatory terms
        tick_inc = torch.arange(1, T + 1, device=device, dtype=tick.dtype)
        ticks = tick + tick_inc.view(T, 1, 1)
        occil = torch.cos(ticks @ self.omegas.unsqueeze(0))
        
        # Apply gating
        values_gated = values * beta
        keys_gated = keys_expanded * gammas_expanded
        s = keys_gated.clone()
        
        # Expand with oscillations
        values_osc = values_gated.unsqueeze(2) * occil.unsqueeze(-1).unsqueeze(-1)
        keys_osc = keys_gated.unsqueeze(2) * occil.unsqueeze(-1).unsqueeze(-1)
        
        # Prepare discount factors
        if self.reset_hidden_on_terminate:
            # terminations: (T, B) -> add dimensions for broadcasting
            term_mask = (1 - terminations.float()).unsqueeze(2).unsqueeze(3)  # (T, B, 1, 1)
            discount_gamma = (1 - gammas_expanded) * term_mask  # gammas_expanded: (T, B, head_num, head_dim*eta)
            discount_beta = (1 - beta) * term_mask  # beta: (T, B, head_num, head_dim)
        else:
            discount_gamma = 1 - gammas_expanded
            discount_beta = 1 - beta
        
        # Compute discounted sums
        if self.optimize_large_batch and B > 1024:
            # Process in chunks for very large batches
            final_keys, final_values, final_s = self._chunked_discounted_sum(
                tilde_k_prev, tilde_v_prev, s_prev,
                keys_osc, values_osc, s,
                discount_gamma, discount_beta, B
            )
        else:
            # Standard discounted sum
            final_keys = discounted_sum(
                tilde_k_prev,
                keys_osc,
                discount_gamma.unsqueeze(2)
            )
            final_values = discounted_sum(
                tilde_v_prev,
                values_osc,
                discount_beta.unsqueeze(2)
            )
            final_s = discounted_sum(
                s_prev,
                s,
                discount_gamma
            )
        
        # Compute attention output
        keys_dot_queries = (final_keys * queries_expanded.unsqueeze(2)).sum(-1)
        kv = (final_values * keys_dot_queries.unsqueeze(-1)).sum(2)
        
        norm = (final_s * queries_expanded).sum(-1) + self.eps
        attn_out = kv / (2 * self.r * norm.unsqueeze(-1))
        
        # Project to d_model
        attn_out = attn_out.view(T, B, self.head_num * self.head_dim)
        attn_out = self.attn_project(attn_out)
        
        # Update memory  
        new_tick = tick + T
        # final_keys shape: (T, B, r, n_heads, eta*d_head)
        # We need: (B, r, n_heads, eta*d_head)
        new_tilde_k = final_keys[-1] if T > 0 else tilde_k_prev
        new_tilde_v = final_values[-1] if T > 0 else tilde_v_prev
        new_s = final_s[-1] if T > 0 else s_prev
        
        return attn_out, (new_tilde_k, new_tilde_v, new_s, new_tick)
    
    def _compute_outer_product(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        T: int,
        B: int
    ) -> torch.Tensor:
        """Compute outer product efficiently using batched operations."""
        TB = T * B
        a_flat = a.reshape(TB * self.head_num, self.head_dim)
        b_flat = b.reshape(TB * self.head_num, self.eta)
        
        result = torch.bmm(
            a_flat.unsqueeze(2),
            b_flat.unsqueeze(1)
        ).reshape(T, B, self.head_num, self.head_dim * self.eta)
        
        return result
    
    def _chunked_discounted_sum(
        self,
        k_prev: torch.Tensor,
        v_prev: torch.Tensor,
        s_prev: torch.Tensor,
        keys_osc: torch.Tensor,
        values_osc: torch.Tensor,
        s: torch.Tensor,
        discount_gamma: torch.Tensor,
        discount_beta: torch.Tensor,
        B: int,
        chunk_size: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process discounted sum in chunks for large batches."""
        T = keys_osc.shape[0]
        
        final_keys_chunks = []
        final_values_chunks = []
        final_s_chunks = []
        
        for i in range(0, B, chunk_size):
            end_idx = min(i + chunk_size, B)
            
            # Process chunk
            k_chunk = discounted_sum(
                k_prev[i:end_idx],
                keys_osc[:, i:end_idx],
                discount_gamma[:, i:end_idx].unsqueeze(2)
            )
            v_chunk = discounted_sum(
                v_prev[i:end_idx],
                values_osc[:, i:end_idx],
                discount_beta[:, i:end_idx].unsqueeze(2)
            )
            s_chunk = discounted_sum(
                s_prev[i:end_idx],
                s[:, i:end_idx],
                discount_gamma[:, i:end_idx]
            )
            
            final_keys_chunks.append(k_chunk)
            final_values_chunks.append(v_chunk)
            final_s_chunks.append(s_chunk)
        
        # Concatenate chunks
        final_keys = torch.cat(final_keys_chunks, dim=1)
        final_values = torch.cat(final_values_chunks, dim=1)
        final_s = torch.cat(final_s_chunks, dim=1)
        
        return final_keys, final_values, final_s
    
    @staticmethod
    def initialize_memory(
        batch_size: int,
        head_num: int,
        head_dim: int,
        eta: int,
        r: int,
        device: torch.device
    ) -> Tuple:
        """Initialize memory state."""
        tilde_k = torch.zeros(batch_size, r, head_num, eta * head_dim, device=device)
        tilde_v = torch.zeros(batch_size, r, head_num, head_dim, device=device)
        s = torch.zeros(batch_size, head_num, eta * head_dim, device=device)
        tick = torch.zeros(batch_size, 1, device=device)
        return (tilde_k, tilde_v, s, tick)


class UnifiedAGaLiTe(nn.Module):
    """
    Full AGaLiTe model with unified implementation.
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int = 4,
        r: int = 8,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_gru_gating: bool = True,
        use_ffc: bool = True,
        optimize_large_batch: bool = True,
        # Feature flags for gradual migration
        optimize_for_speed: bool = False,  # Reduces eta/r when True
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_head = d_head
        self.d_ffc = d_ffc
        self.n_heads = n_heads
        
        # Optionally reduce parameters for speed
        if optimize_for_speed:
            self.eta = min(eta, 2)
            self.r = min(r, 4)
        else:
            self.eta = eta
            self.r = r
        
        # Input embedding
        self.input_embed = nn.Linear(d_model, d_model)
        
        # Create layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                UnifiedAGaLiTeLayer(
                    d_model=d_model,
                    head_num=n_heads,
                    head_dim=d_head,
                    d_ffc=d_ffc,
                    eta=self.eta,
                    r=self.r,
                    reset_hidden_on_terminate=reset_on_terminate,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    use_gru_gating=use_gru_gating,
                    use_ffc=use_ffc,
                    optimize_large_batch=optimize_large_batch,
                )
            )
        
        # Initialize input embedding
        nn.init.orthogonal_(self.input_embed.weight, gain=math.sqrt(2))
        nn.init.constant_(self.input_embed.bias, 0)
    
    def forward(
        self,
        inputs: torch.Tensor,
        terminations: torch.Tensor,
        memory: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through all layers.
        
        Args:
            inputs: (T, B, d_model) or (B, d_model) input tensor
            terminations: (T, B) or (B,) termination signals
            memory: Dictionary of memory states for each layer
        
        Returns:
            output: (T, B, d_model) or (B, d_model) output tensor
            new_memory: Updated memory states
        """
        # Handle both (T, B, d_model) and (B, d_model) inputs
        squeeze_output = False
        if inputs.dim() == 2:
            # Inference mode: (B, d_model) -> (1, B, d_model)
            inputs = inputs.unsqueeze(0)
            if terminations.dim() == 1:
                terminations = terminations.unsqueeze(0)
            squeeze_output = True
        
        # Embed input
        x = self.input_embed(inputs)
        x = F.relu(x)
        
        new_memory = {}
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_key = f"layer_{i + 1}"  # Match legacy 1-indexed naming
            residual = x if i > 0 else None
            
            x, layer_memory = layer(
                x, terminations, memory[layer_key], residual
            )
            new_memory[layer_key] = layer_memory
        
        # Squeeze output if input was 2D
        if squeeze_output:
            x = x.squeeze(0)
        
        return x, new_memory
    
    def initialize_memory(
        self,
        batch_size: int,
        device: torch.device
    ) -> Dict:
        """Initialize memory for all layers."""
        memory = {}
        for i in range(self.n_layers):
            memory[f"layer_{i + 1}"] = UnifiedAGaLiTeLayer.initialize_memory(
                batch_size, self.n_heads, self.d_head,
                self.eta, self.r, device
            )
        return memory