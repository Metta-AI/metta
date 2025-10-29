# Running NCCL Tests on SkyPilot

NCCL tests validate GPU communication infrastructure before training. They should be run:

- Before starting large/expensive training runs
- After changing cloud provider, region, or instance type
- When investigating mysterious training failures or hangs
- After NCCL or GPU driver updates

## Usage

NCCL tests use the **same infrastructure** as training jobs (defined in `job.yaml`), ensuring you're testing the exact
configuration that will be used for training.

### Launch NCCL tests on the cluster

```bash
# Single node (uses default GPU configuration from job.yaml)
./devops/skypilot/launch.py --tool devops.skypilot.tools.nccl run=nccl_diag_$(date +%Y%m%d_%H%M%S)

# Multi-node test (2 nodes, 4 GPUs each)
./devops/skypilot/launch.py --tool devops.skypilot.tools.nccl --nodes 2 --gpus 4 run=nccl_diag_multinode

# 8 GPU test
./devops/skypilot/launch.py --tool devops.skypilot.tools.nccl --gpus 8 run=nccl_diag_8gpu

# Skip using spot instances (for faster startup)
./devops/skypilot/launch.py --tool devops.skypilot.tools.nccl --no-spot run=nccl_diag_ondemand
```

### Check test results

```bash
# View job logs
sky jobs logs <job_id>

# Or check the web dashboard
# https://skypilot-api.softmax-research.net/
```

## What it tests

- **Point-to-point bandwidth**: GPU-to-GPU communication speed between pairs
- **All-reduce bandwidth**: Collective operation performance across all GPUs
- **GPU topology**: NVLink connectivity and NUMA configuration
- **System diagnostics**: NCCL version, CUDA version, driver info

## Configuration

NCCL tests use the same `job.yaml` configuration as training jobs, so you're always testing on the exact infrastructure
that will be used for training. There's no separate configuration to maintain - just modify `job.yaml` and both training
and NCCL tests will use the updated settings.

## Exit codes

- `0`: All tests passed ✅
- `1`: Some tests failed ❌ (check logs for details)

## Example output

When tests pass, you'll see output like:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                      NCCL BANDWIDTH BENCHMARKS                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

  🔊 P2P BANDWIDTH (Rank 0 → Rank 1):
    Message Size : 64 MB
    Bandwidth    : 23.45 GB/s
    Time         : 2.73 ms

  🔊 ALLREDUCE BANDWIDTH:
    Size (MB)    Time (ms)    Bandwidth (GB/s)
    ------------ ------------ ---------------
    1            0.15         6.67
    4            0.45         8.89
    16           1.23         13.01
    64           4.56         14.04

  🚀 Peak Allreduce: 14.04 GB/s at 64MB

╔═══════════════════════════════════════════════════════════════════════════╗
║                             TEST SUMMARY                                   ║
║                                                                           ║
║ Basic Connectivity        : ✔ PASSED                                     ║
║ P2P Bandwidth Test        : ✔ PASSED                                     ║
║ Allreduce Bandwidth Test  : ✔ PASSED                                     ║
║                                                                           ║
║ Overall: ✔ All ranks passed                                              ║
║                                                                           ║
║ ✔ ALL TESTS PASSED! ✔                                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
```
