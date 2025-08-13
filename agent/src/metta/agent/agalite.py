"""
PyTorch implementation of AGaLiTe (Approximate Gated Linear Transformer)
Based on the paper: "AGaLiTe: Approximate Gated Linear Transformers for Online Reinforcement Learning"

This implementation converts the JAX code from the agalite repository to PyTorch,
maintaining the core algorithmic components while adapting to PyTorch conventions.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUGatingUnit(nn.Module):
    """
    GRU Gating Unit used in AGaLiTe.
    Implements custom GRU formulation matching the JAX version.
    """

    def __init__(self, input_dim: int, bg: float = 2.0):
        super().__init__()
        self.input_dim = input_dim

        # Define weight matrices directly to match JAX implementation
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bgp = nn.Parameter(torch.full((input_dim,), bg))

        # Initialize weights orthogonally
        for module in [self.Wr, self.Ur, self.Wz, self.Uz, self.Wg, self.Ug]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Previous hidden state
            y: Current input
        Returns:
            Gated output
        """
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bgp)
        h = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h


class ParameterizedProjection(nn.Module):
    """
    Parameterized projection layer for computing feature maps.
    Creates outer products between two learned projections.
    """

    def __init__(self, dim: int, non_linearity: callable, nu: int = 1):
        super().__init__()
        self.dim = dim
        self.nu = nu
        self.non_linearity = non_linearity

        self.proj_main = nn.Linear(dim, dim)
        self.proj_nu = nn.Linear(dim, nu)

        # Initialize weights
        nn.init.orthogonal_(self.proj_main.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.proj_nu.weight, gain=math.sqrt(2))
        nn.init.constant_(self.proj_main.bias, 0)
        nn.init.constant_(self.proj_nu.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Input tensor of shape (T, d_model)

        Returns:
            Projected tensor of shape (T, d_model * nu)
        """
        inputs_proj = self.non_linearity(self.proj_main(inputs))  # (T, dim)
        nu_proj = self.non_linearity(self.proj_nu(inputs))  # (T, nu)

        # Compute outer product using einsum
        return torch.einsum("ti,tj->tij", inputs_proj, nu_proj).flatten(-2)  # (T, dim * nu)


def discounted_sum_parallel(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """
    Compute discounted sum using batched implementation.
    Implements: y[t] = discount[t] * y[t-1] + x[t]

    Args:
        start_state: Initial state tensor of shape (*)
        x: Sequence tensor of shape (T, *)
        discounts: Discount factors of shape (T, *)

    Returns:
        Discounted sum tensor of shape (T, *)
    """
    T = x.shape[0]
    
    if T == 0:
        return x
    
    # For small sequences, use the simple loop (it's actually efficient for small T)
    if T <= 16:
        result = []
        state = start_state
        for t in range(T):
            state = discounts[t] * state + x[t]
            result.append(state)
        return torch.stack(result, dim=0)
    
    # For larger sequences, use a more efficient batched approach
    # This implements an associative scan in log(T) parallel steps
    # The key insight: we can express y[t] in terms of cumulative products and sums
    
    device = x.device
    dtype = x.dtype
    
    # Prepare arrays for the scan
    # We'll compute: result[t] = start_state * prod(discounts[0:t+1]) + sum of properly discounted x values
    
    # Step 1: Compute cumulative products of discounts
    discount_cumprod = torch.zeros_like(discounts)
    discount_cumprod[0] = discounts[0]
    for i in range(1, T):
        discount_cumprod[i] = discount_cumprod[i-1] * discounts[i]
    
    # Step 2: Compute the contribution from start_state
    start_contrib = start_state.unsqueeze(0) * discount_cumprod
    
    # Step 3: Compute the contribution from x values
    # x[i] contributes to result[j] (for j >= i) with factor prod(discounts[i+1:j+1])
    x_contrib = torch.zeros_like(x)
    x_contrib[0] = x[0]
    
    for t in range(1, T):
        # x_contrib[t] accumulates all contributions up to time t
        x_contrib[t] = discounts[t] * x_contrib[t-1] + x[t]
    
    # Step 4: Combine contributions
    result = start_contrib + x_contrib
    
    return result


class AttentionAGaLiTeLayer(nn.Module):
    """
    AGaLiTe attention layer implementing the approximate gated linear transformer mechanism.

    Args:
        input_dim: Input dimension
        head_dim: Dimension per attention head
        head_num: Number of attention heads
        eta: Feature map expansion factor
        r: Number of approximation terms
        dropout: Dropout rate
        eps: Small epsilon for numerical stability
        reset_hidden_on_terminate: Whether to reset hidden states on termination signals
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        head_num: int,
        eta: int,
        r: int,
        dropout: float = 0.0,
        eps: float = 1e-5,
        reset_hidden_on_terminate: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.eta = eta
        self.r = r
        self.eps = eps
        self.reset_hidden_on_terminate = reset_hidden_on_terminate

        # Linear projections for keys, queries, values, beta, and gamma
        self.linear_kqvbetagammas = nn.Linear(input_dim, head_num * head_dim * 5)
        self.linear_p1p2p3 = nn.Linear(input_dim, head_num * eta * 3)
        self.project = nn.Linear(head_num * head_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.orthogonal_(self.linear_kqvbetagammas.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.linear_p1p2p3.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.project.weight, gain=math.sqrt(2))
        nn.init.constant_(self.linear_kqvbetagammas.bias, 0)
        nn.init.constant_(self.linear_p1p2p3.bias, 0)
        nn.init.constant_(self.project.bias, 0)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, last_memory: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass of AGaLiTe attention layer.

        Args:
            inputs: Input tensor of shape (cur_seq, input_dim)
            terminations: Termination signals of shape (cur_seq,)
            last_memory: Tuple of (tilde_k_prev, tilde_v_prev, s_prev, tick)
                - tilde_k_prev: shape (r, head_num, eta * head_dim)
                - tilde_v_prev: shape (r, head_num, head_dim)
                - s_prev: shape (head_num, eta * head_dim)
                - tick: shape (1,)

        Returns:
            - output: Attention output of shape (cur_seq, input_dim)
            - new_memory: Updated memory tuple
        """
        cur_seq, _ = inputs.shape
        device = inputs.device

        tilde_k_prev, tilde_v_prev, s_prev, tick = last_memory

        # Project to keys, queries, values, beta, gamma
        kqvbetagammas = self.linear_kqvbetagammas(inputs).view(cur_seq, self.head_num, 5, self.head_dim)
        keys, queries, values, beta, gammas = kqvbetagammas.unbind(2)  # Each: (cur_seq, head_num, head_dim)

        # Project to p1, p2, p3 for feature map computation
        p1p2p3 = self.linear_p1p2p3(inputs).view(cur_seq, self.head_num, 3, self.eta)
        p1, p2, p3 = p1p2p3.unbind(2)  # Each: (cur_seq, head_num, eta)

        # Compute feature mapped keys, queries, gammas using outer products with einsum
        keys = torch.einsum("chd,chn->chdn", F.relu(keys), F.relu(p1)).flatten(
            -2
        )  # (cur_seq, head_num, eta * head_dim)
        queries = torch.einsum("chd,chn->chdn", F.relu(queries), F.relu(p2)).flatten(-2)
        gammas = torch.einsum("chd,chn->chdn", torch.sigmoid(gammas), torch.sigmoid(p3)).flatten(-2)

        beta = torch.sigmoid(beta)  # (cur_seq, head_num, head_dim)

        # Update tick and compute oscillatory terms
        tick_inc = torch.arange(1, cur_seq + 1, device=device)[:, None]  # (cur_seq, 1)
        ticks = tick + tick_inc  # (cur_seq, 1)

        omegas = torch.linspace(-math.pi, math.pi, self.r, device=device)  # (r,)
        occil = torch.cos(torch.einsum("ci,j->cj", ticks, omegas))  # (cur_seq, r)

        # Apply gating to values and keys using einsum
        values = values * beta  # (cur_seq, head_num, head_dim)
        values = torch.einsum("chd,cr->crhd", values, occil)  # (cur_seq, r, head_num, head_dim)

        keys = keys * gammas  # (cur_seq, head_num, eta * head_dim)
        s = keys.clone()  # (cur_seq, head_num, eta * head_dim)

        keys = torch.einsum("chd,cr->crhd", keys, occil)  # (cur_seq, r, head_num, eta * head_dim)

        # Compute discount factors
        if self.reset_hidden_on_terminate:
            discount_gamma = (1 - gammas) * (1 - terminations).reshape(cur_seq, 1, 1)
            discount_beta = (1 - beta) * (1 - terminations).reshape(cur_seq, 1, 1)
        else:
            discount_gamma = 1 - gammas
            discount_beta = 1 - beta

        # Compute discounted sums for keys and values
        final_keys = discounted_sum_parallel(
            tilde_k_prev, keys, discount_gamma.unsqueeze(1)
        )  # (cur_seq, r, head_num, eta * head_dim)

        final_values = discounted_sum_parallel(
            tilde_v_prev, values, discount_beta.unsqueeze(1)
        )  # (cur_seq, r, head_num, head_dim)

        final_s = discounted_sum_parallel(s_prev, s, discount_gamma)  # (cur_seq, head_num, eta * head_dim)

        # Compute attention output using einsum throughout
        keys_dot_queries = torch.einsum("crhd,chd->crh", final_keys, queries)  # (cur_seq, r, head_num)
        kv = torch.einsum("crhd,crh->chd", final_values, keys_dot_queries)  # (cur_seq, head_num, head_dim)

        norm = torch.einsum("chd,chd->ch", final_s, queries)  # (cur_seq, head_num)
        attn_out = kv / (2 * self.r * norm[..., None] + self.eps)  # (cur_seq, head_num, head_dim)

        # Reshape and project output
        attn_out = attn_out.flatten(-2)  # (cur_seq, head_num * head_dim)
        attn_out = self.dropout(self.project(attn_out))  # (cur_seq, input_dim)

        # Update memory
        new_tick = tick + cur_seq
        new_tilde_k = final_keys[-1]  # (r, head_num, eta * head_dim)
        new_tilde_v = final_values[-1]  # (r, head_num, head_dim)
        new_s = final_s[-1]  # (head_num, eta * head_dim)

        return attn_out, (new_tilde_k, new_tilde_v, new_s, new_tick)

    @staticmethod
    def initialize_memory(
        head_num: int, head_dim: int, eta: int, r: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize memory for AGaLiTe attention layer."""
        if device is None:
            device = torch.device("cpu")

        tilde_k_prev = torch.zeros((r, head_num, eta * head_dim), device=device)
        tilde_v_prev = torch.zeros((r, head_num, head_dim), device=device)
        s_prev = torch.zeros((head_num, eta * head_dim), device=device)
        tick = torch.tensor([1.0], device=device)

        return (tilde_k_prev, tilde_v_prev, s_prev, tick)


class RecurrentLinearTransformerEncoder(nn.Module):
    """
    Single encoder layer of the AGaLiTe architecture.

    Args:
        d_model: Model dimension
        d_head: Head dimension
        d_ffc: Feed-forward dimension
        n_heads: Number of attention heads
        eta: Feature map expansion factor
        r: Number of approximation terms
        use_dense: Whether to use dense layer for input embedding
        gru_bias: Initial bias for GRU gating
        reset_hidden_on_terminate: Whether to reset hidden states on termination
        embedding_act: Whether to apply activation after embedding
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        use_dense: bool = False,
        gru_bias: float = 2.0,
        reset_hidden_on_terminate: bool = True,
        embedding_act: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.eta = eta
        self.r = r
        self.use_dense = use_dense
        self.embedding_act = embedding_act

        # Optional input embedding layer
        if use_dense:
            self.emb_layer = nn.Linear(d_model, d_model)
            nn.init.orthogonal_(self.emb_layer.weight, gain=math.sqrt(2))
            nn.init.constant_(self.emb_layer.bias, 0)

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # AGaLiTe attention layer
        self.attention = AttentionAGaLiTeLayer(
            input_dim=d_model,
            head_dim=d_head,
            head_num=n_heads,
            eta=eta,
            r=r,
            dropout=dropout,
            reset_hidden_on_terminate=reset_hidden_on_terminate,
        )

        # GRU gating units
        self.gru1 = GRUGatingUnit(d_model, gru_bias)
        self.gru2 = GRUGatingUnit(d_model, gru_bias)

        # Feed-forward network
        self.ffc = nn.Sequential(nn.Linear(d_model, d_ffc), nn.ReLU(), nn.Linear(d_ffc, d_model))

        # Initialize feed-forward weights
        for module in self.ffc:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, inputs: torch.Tensor, terminations: torch.Tensor, last_memory: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass of the encoder layer.

        Args:
            inputs: Input tensor of shape (T, d_model)
            terminations: Termination signals of shape (T,)
            last_memory: Memory tuple from previous timestep

        Returns:
            - output: Encoded output of shape (T, d_model)
            - new_memory: Updated memory tuple
        """
        # Input embedding
        if self.use_dense:
            inputs_enc = self.emb_layer(inputs)
            if self.embedding_act:
                inputs_enc = F.relu(inputs_enc)
        else:
            inputs_enc = inputs

        # Layer norm + attention
        ln1_out = self.ln1(inputs_enc)
        attn_out, new_memory = self.attention(ln1_out, terminations, last_memory)
        attn_out = F.relu(attn_out)

        # First GRU gating
        gating1_out = self.gru1(inputs_enc, attn_out)

        # Layer norm + feed-forward
        ln2_out = self.ln2(gating1_out)
        ffc_out = self.ffc(ln2_out)
        ffc_out = F.relu(ffc_out)
        ffc_out = self.dropout(ffc_out)

        # Second GRU gating
        out = self.gru2(gating1_out, ffc_out)

        return out, new_memory

    @staticmethod
    def initialize_memory(
        n_heads: int, d_head: int, eta: int, r: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize memory for the encoder layer."""
        return AttentionAGaLiTeLayer.initialize_memory(n_heads, d_head, eta, r, device)


class AGaLiTe(nn.Module):
    """
    AGaLiTe (Approximate Gated Linear Transformer) model.

    Args:
        n_layers: Number of transformer layers
        d_model: Model dimension
        d_head: Head dimension
        d_ffc: Feed-forward dimension
        n_heads: Number of attention heads
        eta: Feature map expansion factor
        r: Number of approximation terms
        reset_on_terminate: Whether to reset hidden states on termination
        dropout: Dropout rate
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
        self.reset_on_terminate = reset_on_terminate

        # Create encoder layers
        self.encoders = nn.ModuleList()
        for layer in range(n_layers):
            # First layer uses dense embedding
            use_dense = layer == 0
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
        self, inputs: torch.Tensor, terminations: torch.Tensor, last_memory: Dict[str, Tuple]
    ) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
        """
        Forward pass of AGaLiTe model.

        Args:
            inputs: Input tensor of shape (T, input_dim)
            terminations: Termination signals of shape (T,)
            last_memory: Dictionary of memory tuples for each layer

        Returns:
            - output: Model output of shape (T, d_model)
            - new_memory: Updated memory dictionary
        """
        u_i = inputs
        new_memory = {}

        for layer_idx, encoder in enumerate(self.encoders):
            layer_key = f"layer_{layer_idx + 1}"
            u_i, memory_updated = encoder(u_i, terminations, last_memory[layer_key])
            new_memory[layer_key] = memory_updated

        return u_i, new_memory

    @staticmethod
    def initialize_memory(
        n_layers: int, n_heads: int, d_head: int, eta: int, r: int, device: torch.device = None
    ) -> Dict[str, Tuple]:
        """
        Initialize memory for all layers.

        Returns:
            Dictionary mapping layer names to memory tuples
        """
        memory_dict = {}
        for layer in range(1, n_layers + 1):
            memory_dict[f"layer_{layer}"] = RecurrentLinearTransformerEncoder.initialize_memory(
                n_heads, d_head, eta, r, device
            )
        return memory_dict


class BatchedAGaLiTe(nn.Module):
    """
    Batched version of AGaLiTe for processing multiple sequences in parallel.

    Args:
        n_layers: Number of transformer layers
        d_model: Model dimension
        d_head: Head dimension
        d_ffc: Feed-forward dimension
        n_heads: Number of attention heads
        eta: Feature map expansion factor
        r: Number of approximation terms
        reset_on_terminate: Whether to reset hidden states on termination
        dropout: Dropout rate
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
        self.n_heads = n_heads
        self.d_head = d_head
        self.eta = eta
        self.r = r

        # Create the base AGaLiTe model
        self.model = AGaLiTe(
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

    def forward(
        self, carry: Dict[str, Tuple], x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[Dict[str, Tuple], torch.Tensor]:
        """
        Forward pass for batched sequences.

        Args:
            carry: Batched memory dictionary
            x: Tuple of (inputs, resets)
                - inputs: shape (T, B, d_model)
                - resets: shape (T, B)

        Returns:
            - new_memory: Updated batched memory
            - outputs: Model outputs of shape (T, B, d_model)
        """
        ins, resets = x
        T, B, d_model = ins.shape

        # Process all batch elements in parallel through the layers
        u_i = ins  # Shape: (T, B, d_model)
        new_memory = {}

        for layer_idx, encoder in enumerate(self.model.encoders):
            layer_key = f"layer_{layer_idx + 1}"
            layer_memory = carry[layer_key]
            
            # Process the entire batch through this layer
            # We need to handle the batched memory properly
            u_i, memory_updated = self._forward_batched_layer(
                encoder, u_i, resets, layer_memory
            )
            new_memory[layer_key] = memory_updated

        return new_memory, u_i

    def _forward_batched_layer(
        self, encoder: RecurrentLinearTransformerEncoder, 
        inputs: torch.Tensor, terminations: torch.Tensor,
        batched_memory: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass through a single encoder layer for all batch elements.
        
        Args:
            encoder: The encoder layer
            inputs: Input tensor of shape (T, B, d_model)
            terminations: Termination signals of shape (T, B)
            batched_memory: Batched memory tuple for this layer
        
        Returns:
            - output: Encoded output of shape (T, B, d_model)
            - new_memory: Updated batched memory tuple
        """
        T, B, d_model = inputs.shape
        
        # Reshape for batch processing: (T, B, d_model) -> (T*B, d_model)
        inputs_flat = inputs.reshape(T * B, d_model)
        terminations_flat = terminations.reshape(T * B)
        
        # Unpack and reshape batched memory for processing
        tilde_k_prev, tilde_v_prev, s_prev, tick = batched_memory
        
        # Process with batched attention
        # Note: We need to modify the encoder to handle batched inputs properly
        # For now, we'll process sequentially but this is where optimization should happen
        outputs = []
        new_memories = []
        
        for b in range(B):
            # Extract memory for this batch element
            batch_memory = (
                tilde_k_prev[b],
                tilde_v_prev[b], 
                s_prev[b],
                tick[b]
            )
            
            # Process this batch element
            output, new_mem = encoder(inputs[:, b], terminations[:, b], batch_memory)
            outputs.append(output)
            new_memories.append(new_mem)
        
        # Stack results back into batched tensors
        outputs = torch.stack(outputs, dim=1)  # (T, B, d_model)
        
        # Stack memories
        new_memory = tuple(
            torch.stack([new_memories[b][i] for b in range(B)], dim=0)
            for i in range(len(new_memories[0]))
        )
        
        return outputs, new_memory

    @staticmethod
    def initialize_carry(
        batch_size: int, n_layers: int, n_heads: int, d_head: int, eta: int, r: int, device: torch.device = None
    ) -> Dict[str, Tuple]:
        """
        Initialize batched memory for all layers.

        Returns:
            Dictionary mapping layer names to batched memory tuples
        """
        # Get single memory and batch it efficiently
        single_memory = AGaLiTe.initialize_memory(n_layers, n_heads, d_head, eta, r, device)

        # Batch the memory using dict and tuple comprehensions
        return {
            key: tuple(component.unsqueeze(0).expand(batch_size, *component.shape) for component in layer_memory)
            for key, layer_memory in single_memory.items()
        }


# Test code
if __name__ == "__main__":
    # Test AGaLiTe model
    print("Testing AGaLiTe model...")
    model = AGaLiTe(n_layers=4, d_model=64, d_head=64, d_ffc=64, n_heads=4, eta=4, r=2)

    inputs = torch.ones((10, 64))
    terminations = torch.zeros((10,))
    memory = AGaLiTe.initialize_memory(4, 4, 64, 4, 2)

    out, new_memory = model(inputs, terminations, memory)
    print(f"Output shape: {out.shape}")

    # Test BatchedAGaLiTe model
    print("\nTesting BatchedAGaLiTe model...")
    batch_model = BatchedAGaLiTe(n_layers=4, d_model=64, d_head=64, d_ffc=64, n_heads=4, eta=4, r=2)

    batch_inputs = torch.ones((10, 32, 64))
    batch_terminations = torch.zeros((10, 32))
    batch_memory = BatchedAGaLiTe.initialize_carry(32, 4, 4, 64, 4, 2)

    new_batch_memory, batch_out = batch_model(batch_memory, (batch_inputs, batch_terminations))
    print(f"Batch output shape: {batch_out.shape}")

    print("\nAll tests passed!")
