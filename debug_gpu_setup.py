#!/usr/bin/env python3
"""Diagnostic script to check GPU setup and batch size calculations."""

import os
import platform
import torch
import multiprocessing

def check_gpu_setup():
    """Check GPU and CUDA setup."""
    print("=== GPU Setup Diagnostics ===")
    print(f"Platform: {platform.system()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print("CUDA is not available - training will use CPU")
        print("\nTo fix on Linux with NVIDIA GPU:")
        print("  1. Check NVIDIA driver: nvidia-smi")
        print("  2. Install CUDA toolkit")
        print("  3. Install PyTorch with CUDA support:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print(f"\nCPU count: {os.cpu_count()}")
    print(f"Multiprocessing CPU count: {multiprocessing.cpu_count()}")
    
    # Check environment variables
    print("\n=== Environment Variables ===")
    cuda_vars = ["CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "LOCAL_RANK"]
    for var in cuda_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
    
    # Calculate rollout workers
    print("\n=== Rollout Workers Calculation ===")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    
    cpu_count = os.cpu_count() or 1
    ideal_workers = (cpu_count // 2) // num_gpus
    
    print(f"CPUs: {cpu_count}")
    print(f"GPUs: {num_gpus}")
    print(f"Ideal workers per GPU: (cpu_count // 2) // num_gpus = ({cpu_count} // 2) // {num_gpus} = {ideal_workers}")
    print(f"Rollout workers (post-dehydration): max(1, {ideal_workers}) = {max(1, ideal_workers)}")
    
    # Pre-dehydration calculation (with power of 2 rounding)
    num_workers_old = 1
    while num_workers_old * 2 <= ideal_workers:
        num_workers_old *= 2
    print(f"Rollout workers (pre-dehydration with power-of-2): {num_workers_old}")
    
    # Batch size calculations
    print("\n=== Batch Size Calculations ===")
    forward_pass_minibatch_target_size = 4096  # default
    num_agents = 24  # typical value
    async_factor = 2  # default
    
    print(f"Forward pass minibatch target size: {forward_pass_minibatch_target_size}")
    print(f"Number of agents: {num_agents}")
    print(f"Async factor: {async_factor}")
    
    for workers in [max(1, ideal_workers), num_workers_old]:
        print(f"\nWith {workers} rollout workers:")
        target_batch_size = forward_pass_minibatch_target_size // num_agents
        if target_batch_size < max(2, workers):
            target_batch_size = workers
        batch_size = (target_batch_size // workers) * workers
        num_envs = batch_size * async_factor
        
        print(f"  Target batch size: {target_batch_size}")
        print(f"  Actual batch size: {batch_size}")
        print(f"  Number of environments: {num_envs}")
        print(f"  Environments per worker: {num_envs // workers if workers > 0 else 0}")

    # Check if PufferLib can use GPU kernels
    print("\n=== PufferLib GPU Support ===")
    try:
        import pufferlib
        print(f"PufferLib version: {pufferlib.__version__ if hasattr(pufferlib, '__version__') else 'unknown'}")
        
        # Check if PufferLib was compiled with CUDA support
        # This is a heuristic - PufferLib might have GPU kernels if CUDA is available
        if torch.cuda.is_available():
            print("PufferLib can potentially use GPU kernels for environment vectorization")
            print("Performance boost expected when CUDA is available")
        else:
            print("PufferLib will use CPU-only implementation")
            print("This can significantly impact training performance")
    except ImportError:
        print("PufferLib not installed")

if __name__ == "__main__":
    check_gpu_setup()