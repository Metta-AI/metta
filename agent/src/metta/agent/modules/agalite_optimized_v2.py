"""
Further optimized AGaLiTe operations with fused computations.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


@torch.jit.script
def fused_discounted_sum_triple(
    start_state_1: torch.Tensor,
    start_state_2: torch.Tensor, 
    start_state_3: torch.Tensor,
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    x_3: torch.Tensor,
    discounts_1: torch.Tensor,
    discounts_2: torch.Tensor,
    discounts_3: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute three discounted sums in a single JIT-compiled function.
    This reduces Python overhead and improves cache locality.
    """
    T = x_1.shape[0]
    if T == 0:
        return x_1, x_2, x_3
    
    # Build all three outputs together
    output_1_list = []
    output_2_list = []
    output_3_list = []
    
    # First step for all three
    prev_1 = discounts_1[0] * start_state_1 + x_1[0]
    prev_2 = discounts_2[0] * start_state_2 + x_2[0]
    prev_3 = discounts_3[0] * start_state_3 + x_3[0]
    
    output_1_list.append(prev_1)
    output_2_list.append(prev_2)
    output_3_list.append(prev_3)
    
    # Process remaining steps
    for t in range(1, T):
        prev_1 = discounts_1[t] * prev_1 + x_1[t]
        prev_2 = discounts_2[t] * prev_2 + x_2[t]
        prev_3 = discounts_3[t] * prev_3 + x_3[t]
        
        output_1_list.append(prev_1)
        output_2_list.append(prev_2)
        output_3_list.append(prev_3)
    
    # Stack all outputs
    return (
        torch.stack(output_1_list, dim=0),
        torch.stack(output_2_list, dim=0),
        torch.stack(output_3_list, dim=0)
    )


def optimized_agalite_forward(
    inputs: torch.Tensor,
    terminations: torch.Tensor,
    tilde_k_prev: torch.Tensor,
    tilde_v_prev: torch.Tensor,
    s_prev: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    s: torch.Tensor,
    discount_gamma_r: torch.Tensor,
    discount_beta_r: torch.Tensor,
    discount_gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized version that calls fused discounted sum.
    
    This should be used in place of three separate discounted_sum calls
    in the AGaLiTe attention layer.
    """
    # Ensure start states have correct dimensions
    if tilde_k_prev.dim() < keys.dim() - 1:
        for _ in range(keys.dim() - 1 - tilde_k_prev.dim()):
            tilde_k_prev = tilde_k_prev.unsqueeze(-1)
    
    if tilde_v_prev.dim() < values.dim() - 1:
        for _ in range(values.dim() - 1 - tilde_v_prev.dim()):
            tilde_v_prev = tilde_v_prev.unsqueeze(-1)
    
    if s_prev.dim() < s.dim() - 1:
        for _ in range(s.dim() - 1 - s_prev.dim()):
            s_prev = s_prev.unsqueeze(-1)
    
    # Call fused function
    return fused_discounted_sum_triple(
        tilde_k_prev, tilde_v_prev, s_prev,
        keys, values, s,
        discount_gamma_r, discount_beta_r, discount_gamma
    )