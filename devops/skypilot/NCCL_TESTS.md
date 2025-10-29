# Running NCCL Tests on SkyPilot

NCCL tests validate GPU communication infrastructure before training.

## When to run

- Before large/expensive training runs
- After changing cloud provider, region, or instance type
- When investigating training failures or hangs
- After NCCL or GPU driver updates

## Usage

NCCL tests use the same `job.yaml` configuration as training jobs, ensuring you're testing the exact infrastructure that
will be used for training.

### Automated test matrix (recommended)

```bash
# Launch tests across multiple configurations (1 node/4 GPUs, 2 nodes/4 GPUs, 4 nodes/4 GPUs)
./devops/skypilot/tests/nccl_test.py launch

# Check test results
./devops/skypilot/tests/nccl_test.py check

# Check with detailed logs
./devops/skypilot/tests/nccl_test.py check -l
```

### Manual single test

```bash
# Single node (uses default GPU configuration)
./devops/skypilot/launch.py --tool devops.skypilot.tools.nccl run=nccl_diag_$(date +%Y%m%d_%H%M%S)

# Multi-node test (2 nodes, 4 GPUs each)
./devops/skypilot/launch.py --tool devops.skypilot.tools.nccl --nodes 2 --gpus 4 run=nccl_diag_multinode

# Check results
sky jobs logs <job_id>
```

## What it tests

- Point-to-point bandwidth: GPU-to-GPU communication speed
- All-reduce bandwidth: Collective operation performance
- GPU topology: NVLink connectivity and NUMA configuration
- System diagnostics: NCCL version, CUDA version, driver info

## Exit codes

- `0`: All tests passed
- `1`: Some tests failed (check logs for details)
