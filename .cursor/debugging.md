# Metta Training Debugging Guide

## Common Issues and Solutions

### 1. MPS Device Error on macOS

**Error**: `AttributeError("module 'torch.mps' has no attribute 'current_device'")`

**Cause**: The system defaults to MPS (Metal Performance Shaders) on macOS when CUDA is requested but not available. PyTorch's MPS support is incomplete and doesn't have all CUDA APIs.

**Solutions**:
- Always specify `device=cpu` on macOS
- Use `+hardware=macbook` which sets device=cpu
- Update user configs to include `device: cpu`

### 2. Training Hangs on macOS

**Cause**: Multiprocessing issues on macOS, especially with fork() safety.

**Solutions**:
- Use serial vectorization: `vectorization=serial`
- Reduce workers: `trainer.num_workers=1`
- Use environment variable: `export PYTORCH_ENABLE_MPS_FALLBACK=1`

### 3. Recommended Commands for macOS

```bash
# Basic training that works on macOS
./tools/train.py +user=cursor run=test_$TEST_ID device=cpu vectorization=serial trainer.num_workers=1

# With puffer initial policy
./tools/train.py +user=cursor run=test_$TEST_ID device=cpu vectorization=serial trainer.num_workers=1 trainer.initial_policy.uri=pytorch://checkpoints/metta-new/metta.pt

# Using hardware config
./tools/train.py +user=cursor run=test_$TEST_ID +hardware=macbook
```

### 4. Debug Mode

To get more information when training hangs:

```bash
# Enable full Hydra error traces
export HYDRA_FULL_ERROR=1

# Enable PyTorch debugging
export TORCH_SHOW_CPP_STACKTRACES=1

# Run with minimal config
./tools/train.py +user=cursor run=debug_test device=cpu vectorization=serial trainer.num_workers=1 trainer.total_timesteps=100 trainer.verbose=true
```

### 5. Check System Status

```bash
# Check if process is running
ps aux | grep train.py

# Check system resources
top -l 1 | head -20

# Check for zombie processes
ps aux | grep defunct
```

### 6. Run Training with Minimal Configuration
```bash
# Create a test ID for easier tracking
export TEST_ID=$(date +%Y%m%d_%H%M%S)

# Run minimal training job
./tools/train.py +user=cursor run=test_$TEST_ID device=cpu vectorization=serial trainer.num_workers=1 trainer.initial_policy.uri=pytorch://checkpoints/metta-new/metta.pt
```
