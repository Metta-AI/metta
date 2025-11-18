## Overview: Performance Checking Experiments

To help you get started with performance checking these experiments, here's what you'll need to do:

### 1. **Initial Setup**

First, ensure your SkyPilot environment is properly configured:

```shell script
# Verify SkyPilot connectivity
uv run sky api info

# Source the shell aliases for convenience
source ./setup_shell.sh
```


### 2. **Running Experiments for Performance Analysis**

Based on the architecture documentation, you'll want to:

#### **Launch Training Experiments**

```shell script
# Basic training run with performance tracking
uv run launch.py recipes.experiment.arena.train run=perfcheck_$(date +%Y%m%d_%H%M%S) \
  trainer.total_timesteps=100000000 \
  --gpus 1 --skip-git-check 

# Multi-GPU performance test
uv run launch.py recipes.experiment.arena.train run=perfcheck_multi_gpu \
  trainer.total_timesteps=100000000 \
  --gpus 4 \
  --nodes 1 --skip-git-check
```


#### **Monitor Job Progress**

```shell script
# List running jobs (using alias from setup_shell.sh)
jj

# View live logs
jl 233797d8

# Stream logs in real-time
uv run sky jobs logs 233797d8 --follow
```


### 3. **Key Performance Metrics to Track**

When running these experiments, you should measure:

- **SPS (Steps Per Second)**: Training throughput
- **GPU Utilization**: Resource efficiency
- **Memory Usage**: Peak and average
- **Wall-clock Time**: Total experiment duration
- **Learning Curves**: Reward progression over time

These metrics are automatically logged to Weights & Biases (WandB) during training.

### 4. **Setting Up Performance Benchmarks**

```shell script
# Run a baseline performance test
./devops/skypilot/launch.py train run=baseline_perf \
  trainer.total_timesteps=10000000 \
  --gpus 1 \
  --confirm  # Preview configuration before launching
```
RESULT: https://skypilot-api.softmax-research.net/dashboard/jobs/8155

```shell script
# Run with different policy architectures for comparison
./devops/skypilot/launch.py train run=perf_vit \
  policy_architecture=latent_attn_small \
  trainer.total_timesteps=10000000

./devops/skypilot/launch.py train run=perf_fast \
  policy_architecture=fast \
  trainer.total_timesteps=10000000
```


### 5. **Analyzing Results**

After experiments complete:

```shell script
# View job completion status
uv run sky jobs queue -a

# Access WandB dashboard to view metrics
# The training logs will include WandB run URLs
```


### 6. **Cost Management Tips**

```shell script
# Set maximum runtime to prevent runaway costs
./devops/skypilot/launch.py train run=perfcheck \
  --max-runtime-hours 2 \
  --gpus 1

# Use spot instances (cheaper but may be interrupted)
./devops/skypilot/launch.py train run=perfcheck_spot \
  --gpus 1  # Spot is default

# Use on-demand for critical experiments
./devops/skypilot/launch.py train run=perfcheck_reliable \
  --no-spot \
  --gpus 1
```


### 7. **Recommended Performance Testing Workflow**

1. **Quick Smoke Test** (5-10 minutes):
```shell script
./devops/skypilot/launch.py train run=smoke_test \
     trainer.total_timesteps=1000000 \
     --max-runtime-hours 0.5
```


2. **Short Performance Run** (30-60 minutes):
```shell script
./devops/skypilot/launch.py train run=short_perf \
     trainer.total_timesteps=10000000 \
     --max-runtime-hours 1
```


3. **Full Performance Baseline** (2-4 hours):
```shell script
./devops/skypilot/launch.py train run=full_baseline \
     trainer.total_timesteps=100000000 \
     --max-runtime-hours 4
```


### 8. **Parallel Performance Testing**

Run multiple experiments simultaneously:

```shell script
# Launch 3 identical runs for statistical confidence
./devops/skypilot/launch.py train run=perf_replicate \
  trainer.total_timesteps=50000000 \
  --copies 3 \
  --max-runtime-hours 2
```


### 9. **Debugging Performance Issues**

If experiments are slower than expected:

```shell script
# Enable PyTorch profiler
./devops/skypilot/launch.py train run=profile_run \
  trainer.total_timesteps=1000000 \
  # Note: Profiling configuration is in TrainerConfig

# Check controller logs for issues
jlc <JOB_ID>

# View detailed resource usage
uv run sky status
```


### Next Steps

To begin your performance investigation:

1. **Read the mp-perfcheck.md file** to understand specific experiments defined there
2. **Run a smoke test** to verify everything works
3. **Launch baseline experiments** with different configurations
4. **Monitor and compare** SPS metrics across runs
5. **Analyze results** using WandB dashboards

Would you like me to help you:
- Read and analyze the specific contents of `mp-perfcheck.md`?
- Create a custom performance testing script?
- Set up automated performance regression testing?
- Analyze existing performance results?