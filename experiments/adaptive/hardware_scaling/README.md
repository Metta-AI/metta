# Hardware Scaling Experiment

This experiment launches one Bayesian sweep per hardware configuration (GPU/node pair) using the existing SweepTool.

## Quick Start

Single sweep (CLI):
```bash
# Create a single sweep for gpus=1, nodes=1
uv run ./tools/run.py experiments.adaptive.hardware_scaling.hw_sweep \
    gpus=1 nodes=1 \
    recipe_module=experiments.recipes.arena_basic_easy_shaped \
    train_entrypoint=train eval_entrypoint=evaluate \
    total_timesteps=300000000 \
    sweep_name=hs_g1n1 \
    wandb.entity=YOUR_ENTITY wandb.project=hardware-scaling \
    dispatcher_type=skypilot \
    max_trials=50 batch_size=4 max_parallel_jobs=4
```

Batch launcher (spawns one subprocess per pair):
```bash
python experiments/adaptive/hardware_scaling/launch.py \
    --wandb-entity YOUR_ENTITY \
    --wandb-project hardware-scaling \
    --gpu-counts 1 2 4 8 \
    --node-counts 1 2 \
    --dispatcher skypilot \
    --max-trials 50 --batch-size 4 --max-parallel-jobs 4
```

For quick local testing, use `--dispatcher local` and reduce `--max-trials`.

## Components

### 1. Hardware-aware Tool (`hw_sweep.py`)
- `hw_sweep`: tiny factory that returns a configured `SweepTool`
- Uses optimized protein settings based on `(gpus, nodes)`
- Keeps all orchestration inside the SweepTool (W&B, scheduling, dispatch)

### 2. Launcher (`launch.py`)
- Simple subprocess launcher over `(gpus, nodes)` pairs
- Builds `sweep_name` and forwards CLI overrides to `tools/run.py`
- Supports `--sequential` and `--delay` for staggered starts

### 3. Optimized Config (`optimized_sweep_config.py`)
- Creates Protein configs with exact default values from codebase
- Key settings:
  - `num_random_samples=0` - Start Bayesian optimization immediately
  - `seed_with_search_center=True` - Start from default values
  - `bptt_horizon=64` - From TrainerConfig
- Computes batch/minibatch constraints from hardware

### 4. Analysis (`analysis.py`)
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

Default configurations (customizable via `launch.py`):
- GPUs: 1, 2, 4, 8 per node
- Nodes: 1, 2
- Only valid pairs are launched (respect 8 GPUs/node max)

## Environment Variables

Required:
- `WANDB_ENTITY`: Your WandB entity (or use --wandb-entity)
- `WANDB_PROJECT`: Project name (default: "hardware-scaling")

Optional:
- `ANTHROPIC_API_KEY`: For auto-fixing linting issues

## Notes

- The previous `simple_sweep_runner.py` has been superseded by `hw_sweep` + `launch.py` for a much simpler flow.
- Use `dispatcher_type=local` for quick tests (SweepTool also supports `local_test=true`, but not needed here).
