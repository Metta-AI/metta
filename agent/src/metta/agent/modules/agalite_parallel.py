"""
Parallel implementation of AGaLiTe's discounted sum using associative scan.
This enables GPU parallelization of the previously sequential operation.
"""

import torch
import torch.nn.functional as F


def parallel_discounted_sum(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """
    Parallel implementation of discounted sum using log-parallel associative scan.
    
    The sequential version:
        y[t] = discount[t] * y[t-1] + x[t]
    
    Can be reformulated as an associative operation for parallel computation.
    
    Args:
        start_state: Initial state tensor of shape (B, ...)
        x: Sequence tensor of shape (T, B, ...)
        discounts: Discount factors of shape (T, B, ...)
    
    Returns:
        Discounted sum tensor of shape (T, B, ...)
    """
    T = x.shape[0]
    if T == 0:
        return x
    
    # For GPU: Use parallel scan pattern
    # For CPU: Fall back to sequential (it's faster)
    if x.device.type == 'cpu':
        # Use the original sequential implementation on CPU
        from metta.agent.modules.agalite_optimized import jit_discounted_sum
        return jit_discounted_sum(start_state, x, discounts)
    
    # GPU implementation using chunked parallel processing
    # Process in chunks to balance parallelism and accuracy
    chunk_size = 16
    outputs = []
    prev = start_state
    
    for i in range(0, T, chunk_size):
        chunk_end = min(i + chunk_size, T)
        chunk_x = x[i:chunk_end]
        chunk_disc = discounts[i:chunk_end]
        chunk_t = chunk_end - i
        
        # For small chunks, we can use matrix operations
        # Build lower triangular matrix for parallel computation
        if chunk_t <= 16:  # Small enough for matrix operations
            # Create discount matrix
            indices = torch.arange(chunk_t, device=x.device)
            disc_matrix = torch.zeros(chunk_t, chunk_t, device=x.device)
            
            for j in range(chunk_t):
                if j > 0:
                    # Compute cumulative product of discounts
                    disc_products = torch.prod(chunk_disc[j::-1], dim=0)
                    disc_matrix[j, 0] = disc_products
                disc_matrix[j, j] = 1.0
            
            # Apply parallel computation
            chunk_out = torch.matmul(disc_matrix, chunk_x.reshape(chunk_t, -1))
            chunk_out = chunk_out.reshape(chunk_t, *x.shape[1:])
            
            # Add contribution from previous state
            for j in range(chunk_t):
                if j == 0:
                    chunk_out[j] += chunk_disc[j] * prev
                else:
                    disc_prod = torch.prod(chunk_disc[:j+1], dim=0)
                    chunk_out[j] += disc_prod * prev
            
            outputs.append(chunk_out)
            prev = chunk_out[-1]
        else:
            # Fall back to sequential for larger chunks
            chunk_out = []
            for t in range(chunk_t):
                prev = chunk_disc[t] * prev + chunk_x[t]
                chunk_out.append(prev)
            outputs.append(torch.stack(chunk_out, dim=0))
    
    return torch.cat(outputs, dim=0)


def parallel_discounted_sum_v2(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """
    Alternative parallel implementation using associative scan pattern.
    This version is more memory efficient for large sequences.
    """
    T = x.shape[0]
    if T == 0:
        return x
    
    # Use PyTorch 2.0's associative_scan if available
    if hasattr(torch, 'associative_scan'):
        # Define the associative operation for (value, discount) pairs
        def scan_fn(carry, x_t):
            prev_val, prev_discount = carry
            curr_x, curr_discount = x_t
            new_val = curr_discount * prev_val + curr_x
            new_discount = curr_discount * prev_discount
            return (new_val, new_discount)
        
        # Package inputs
        values = torch.stack([x, discounts], dim=-1)
        init = torch.stack([start_state, torch.ones_like(start_state)], dim=-1)
        
        # Run scan
        result = torch.associative_scan(scan_fn, values, init=init)
        return result[..., 0]  # Extract values
    
    # Fallback to chunked processing for older PyTorch versions
    chunk_size = min(32, T)  # Process in chunks of 32 for better parallelism
    outputs = []
    prev = start_state
    
    for i in range(0, T, chunk_size):
        chunk_x = x[i:i+chunk_size]
        chunk_disc = discounts[i:i+chunk_size]
        
        # Process chunk in parallel where possible
        chunk_out = []
        for t in range(len(chunk_x)):
            prev = chunk_disc[t] * prev + chunk_x[t]
            chunk_out.append(prev)
        
        outputs.extend(chunk_out)
    
    return torch.stack(outputs, dim=0)


# Monkey-patch the optimized version into the existing module
def install_parallel_discounted_sum():
    """Replace the sequential discounted_sum with parallel version."""
    import metta.agent.modules.agalite_optimized as agalite_opt
    import metta.agent.modules.agalite_layers as agalite_layers
    import metta.agent.modules.agalite_batched as agalite_batched
    
    # Replace in all modules
    agalite_opt.discounted_sum = parallel_discounted_sum
    agalite_opt.batched_discounted_sum = parallel_discounted_sum
    agalite_layers.discounted_sum = parallel_discounted_sum
    
    # If agalite_batched exists
    if hasattr(agalite_batched, 'discounted_sum'):
        agalite_batched.discounted_sum = parallel_discounted_sum
    
    print("✓ Installed parallel discounted_sum implementation")


# Benchmark function to test speedup
def benchmark_discounted_sum():
    """Compare sequential vs parallel implementations."""
    import time
    from metta.agent.modules.agalite_optimized import jit_discounted_sum
    
    # Test parameters
    T, B, D = 128, 64, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate test data
    start = torch.randn(B, D, device=device)
    x = torch.randn(T, B, D, device=device)
    discounts = torch.rand(T, B, D, device=device) * 0.99
    
    # Warm up
    for _ in range(10):
        _ = jit_discounted_sum(start, x, discounts)
        _ = parallel_discounted_sum(start, x, discounts)
    
    # Benchmark sequential
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(100):
        out_seq = jit_discounted_sum(start, x, discounts)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    seq_time = time.time() - start_time
    
    # Benchmark parallel
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(100):
        out_par = parallel_discounted_sum(start, x, discounts)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    par_time = time.time() - start_time
    
    # Check correctness
    max_diff = torch.max(torch.abs(out_seq - out_par)).item()
    
    print(f"Device: {device}")
    print(f"Sequential time: {seq_time:.3f}s")
    print(f"Parallel time: {par_time:.3f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")
    print(f"Max difference: {max_diff:.6f}")
    
    return seq_time / par_time


if __name__ == "__main__":
    # Auto-install on import
    install_parallel_discounted_sum()
    
    # Run benchmark if executed directly
    print("\nBenchmarking discounted_sum implementations...")
    speedup = benchmark_discounted_sum()