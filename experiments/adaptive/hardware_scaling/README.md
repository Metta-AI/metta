# Hardware Scaling Experiment

This experiment tests different hardware configurations (GPUs/nodes) to understand scaling laws and find optimal hyperparameters for each configuration.

## Quick Start

```bash
# Run experiment with default settings
python experiments/adaptive/hardware_scaling/simple_sweep_runner.py \
    --wandb-entity YOUR_ENTITY \
    --wandb-project hardware-scaling \
    --gpu-counts 1 2 4 8 \
    --node-counts 1 2 4 \
    --max-trials 50 \
    --total-timesteps 300000000

# Run with fewer trials for testing
python experiments/adaptive/hardware_scaling/simple_sweep_runner.py \
    --wandb-entity YOUR_ENTITY \
    --wandb-project hardware-scaling-test \
    --gpu-counts 1 2 \
    --node-counts 1 \
    --max-trials 10 \
    --total-timesteps 10000000 \
    --batch-size 2
```

## Components

### 1. Simple Sweep Runner (`simple_sweep_runner.py`)
- Launches independent Protein sweeps for each hardware configuration
- Uses optimized settings with exact defaults from codebase
- Each sweep finds optimal hyperparameters for its hardware config

### 2. Optimized Config (`optimized_sweep_config.py`)
- Creates Protein configs with exact default values from codebase
- Key settings:
  - `num_random_samples=0` - Start Bayesian optimization immediately
  - `seed_with_search_center=True` - Start from default values
  - `bptt_horizon=64` - Correct default from TrainerConfig
- Calculates minimum batch sizes based on hardware constraints

### 3. Analysis (`analysis.py`)
- Analyzes results after sweeps complete
- Fits scaling laws for samples, time, and cost
- Calculates Figure of Merit (FOM) for different priorities
- Recommends optimal configurations

## Key Features

### Optimized Hyperparameter Defaults
All hyperparameters start from exact codebase defaults:
- Learning rate: 0.001153637
- PPO clip coefficient: 0.264407
- GAE lambda: 0.891477
- VF clip coefficient: 0.1
- Entropy coefficient: 0.01
- VF coefficient: 0.897619

### Batch Size Constraints
Minimum batch size calculation:
```
min_batch_size = num_envs * num_agents * bptt_horizon * total_gpus
```
Where:
- `num_envs` = Environment instances per GPU
- `num_agents` = 24 (agents per environment)
- `bptt_horizon` = 64 (sequence length)
- `total_gpus` = gpus * nodes

## Analysis

After experiments complete, run analysis:

```bash
# Using wandb data
python -c "
from experiments.adaptive.hardware_scaling.analysis import HardwareScalingAnalyzer
import wandb

api = wandb.Api()
runs = api.runs('YOUR_ENTITY/hardware-scaling')

analyzer = HardwareScalingAnalyzer(target_performance=0.9)
df = analyzer.analyze_wandb_runs(runs)
report = analyzer.generate_report(df)
print(report)

# Save results
df.to_csv('hardware_scaling_results.csv', index=False)
"
```

## Expected Outputs

1. **Per-Hardware Optimal Hyperparameters**: Best configuration for each GPU/node combination
2. **Scaling Laws**: Power law relationships for samples, time, and cost
3. **FOM Analysis**: Trade-offs between sample efficiency, time, and cost
4. **Recommendations**: Best hardware configs for different priorities:
   - Research (sample efficiency)
   - Development (time efficiency)
   - Budget (cost efficiency)
   - Balanced (all factors)

## Hardware Configurations Tested

Default configurations (can be customized via CLI):
- GPUs: 1, 2, 4, 8 per node
- Nodes: 1, 2, 4
- Valid combinations respect 8 GPUs/node maximum

## Environment Variables

Required:
- `WANDB_ENTITY`: Your WandB entity (or use --wandb-entity)
- `WANDB_PROJECT`: Project name (default: "hardware-scaling")

Optional:
- `ANTHROPIC_API_KEY`: For auto-fixing linting issues