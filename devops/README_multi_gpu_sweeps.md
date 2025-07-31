# Multi-GPU Sweep Runs

This document explains how to run sweep training with proper GPU isolation for distributed training.

## Overview

The sweep infrastructure now supports proper GPU configuration for distributed training, allowing you to:

1. Run single sweep runs with multiple GPUs (distributed training)
2. Run multiple independent sweep runs on different GPU sets
3. Ensure proper CUDA device visibility for each worker in distributed training

## Implementation

### 1. train.sh Modifications

The `devops/train.sh` script now supports GPU configuration through the `GPU_LIST` environment variable:

- If `GPU_LIST` is set (e.g., "0,1,2,3"), it sets `CUDA_VISIBLE_DEVICES` to that list
- This ensures each worker process only sees the GPUs assigned to its training run
- The script automatically counts the number of GPUs from the list

### 2. sweep_rollout.py Modifications

The `tools/sweep_rollout.py` script now accepts a `gpu_list` parameter:

- Pass `gpu_list=0,1,2,3` to assign specific GPUs to a sweep run
- The GPU list is passed to train.sh via the `GPU_LIST` environment variable

## Usage Examples

### Single Sweep Run with Multiple GPUs (Distributed Training)

Run a sweep using 4 GPUs for distributed training:

```bash
./devops/sweep.sh run=my_sweep gpu_list=0,1,2,3
```

Run a sweep using GPUs 4-7:

```bash
./devops/sweep.sh run=my_sweep gpu_list=4,5,6,7
```

### Multiple Independent Sweep Runs

Launch multiple sweep runs, each using different GPU sets:

```bash
# Terminal 1 - Run 1 uses GPUs 0,1
./devops/sweep.sh run=my_sweep.run1 gpu_list=0,1

# Terminal 2 - Run 2 uses GPUs 2,3
./devops/sweep.sh run=my_sweep.run2 gpu_list=2,3

# Terminal 3 - Run 3 uses GPUs 4,5
./devops/sweep.sh run=my_sweep.run3 gpu_list=4,5

# Terminal 4 - Run 4 uses GPUs 6,7
./devops/sweep.sh run=my_sweep.run4 gpu_list=6,7
```

### Automated Multi-Run Launch

Use the helper script to launch multiple runs automatically:

```bash
# Launch 4 runs, each using 2 GPUs
./devops/sweep_multi_gpu.sh my_sweep 2 4

# Launch 2 runs, each using 4 GPUs (8 total GPUs)
./devops/sweep_multi_gpu.sh my_sweep 4 2 8

# With additional parameters
./devops/sweep_multi_gpu.sh my_sweep 2 4 8 trainer.total_timesteps=1000000
```

## How It Works

### Distributed Training Fix

When `torchrun` launches multiple worker processes for distributed training:

1. Each worker process inherits the `CUDA_VISIBLE_DEVICES` setting
2. Worker 0 sees GPU 0 as device 0, Worker 1 sees GPU 1 as device 0, etc.
3. This prevents device confusion where all workers try to use the same physical GPU

### Example

For `gpu_list=4,5,6,7`:

- `CUDA_VISIBLE_DEVICES=4,5,6,7` is set before launching torchrun
- Worker 0 (LOCAL_RANK=0) uses GPU 4 (but sees it as cuda:0)
- Worker 1 (LOCAL_RANK=1) uses GPU 5 (but sees it as cuda:1)
- Worker 2 (LOCAL_RANK=2) uses GPU 6 (but sees it as cuda:2)
- Worker 3 (LOCAL_RANK=3) uses GPU 7 (but sees it as cuda:3)

## Important Notes

1. **GPU Indexing**: GPU indices are 0-based (0,1,2,3,4,5,6,7 for an 8-GPU system)

2. **Distributed Training**: Each worker automatically selects the correct GPU based on its LOCAL_RANK

3. **Resource Planning**: Ensure GPU assignments don't overlap between concurrent runs

4. **Memory**: Each GPU needs enough memory for its portion of the distributed training

## Monitoring

Monitor GPU usage and assignments:

```bash
# Check GPU usage
nvidia-smi -l 1

# Check CUDA visibility for a process
ps aux | grep train.py
cat /proc/<PID>/environ | tr '\0' '\n' | grep CUDA

# Monitor specific runs
tail -f $DATA_DIR/sweep/my_sweep.run1/logs/*
```

## Troubleshooting

1. **Device Errors**: If you see "CUDA error: invalid device ordinal", check that your gpu_list contains valid GPU
   indices

2. **All Workers on Same GPU**: This was the original problem - now fixed by setting CUDA_VISIBLE_DEVICES before
   torchrun

3. **Port Conflicts**: Each sweep run uses the same MASTER_PORT (12345) by default, but this is fine as long as they're
   using different GPU sets
