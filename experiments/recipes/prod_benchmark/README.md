# Production Benchmark Pipeline

This directory contains tools for benchmarking agent architectures with statistical rigor.

## Quick Start

```bash
# 1. Launch 15 training runs with different seeds
./run_icl_benchmark.sh --architecture vit_reset --curriculum terrain_2

# 2. After runs complete, analyze variance to determine stability
python variance.py run1 run2 run3 ... run15 \
    --metric overview/reward \
    --threshold 0.05 \
    --output variance_plot.png

# 3. Compare architectures with statistical analysis
python analysis.py --control-run-ids run1 run2 ... --candidate-run-ids run16 run17 ...
```

## Baseline Control Protocol

1. **Record git hash** - Document the exact codebase version
   ```bash
   git rev-parse HEAD > benchmark_git_hash.txt
   ```

2. **Check for trainer changes** - Diff since last benchmark
   ```bash
   git diff <last-benchmark-commit> -- metta/rl/
   ```

3. **Hyperparameter sweep** - Tune default PPO loss on vit_reset with ICL recipe
   - Use W&B sweeps or manual grid search
   - Document final hyperparameters

4. **Pilot power analysis** - Determine required sample size
   ```bash
   # Run initial 15 training runs
   ./run_icl_benchmark.sh --architecture vit_reset --base-run-name pilot_baseline

   # Analyze variance to determine if N=15 is sufficient
   python variance.py <15-run-ids> --threshold 0.05
   ```
   - If variance > 5% at N=15, run more seeds
   - Recommended: Start with 15, extend to 20-30 if needed

## Benchmarking a New Intervention

### 1. Hyperparameter Sweep

- Tune your new architecture on the same ICL recipe
- **Critical**: Align model capacity (parameter count) with baseline
- Document all hyperparameter changes

### 2. Production Runs with Paired Seeds

```bash
# Launch control runs (baseline architecture)
./run_icl_benchmark.sh \
    --architecture vit_reset \
    --curriculum terrain_2 \
    --base-run-name control_vit_reset_20250105

# Launch treatment runs (your new architecture)
./run_icl_benchmark.sh \
    --architecture trxl \
    --curriculum terrain_2 \
    --base-run-name treatment_trxl_20250105
```

**Important**: Use the same seeds for both control and treatment (paired design reduces variance)

### 3. Statistical Analysis

```bash
# Compare architectures with bootstrap confidence intervals
python analysis.py \
    --pairs '{"control": "control_run1", "candidate": "treatment_run1"}' \
           '{"control": "control_run2", "candidate": "treatment_run2"}' \
    --metric overview/reward \
    --summary.type auc \
    --bootstrap.n_resamples 10000 \
    --output.csv_path results.csv
```

### 4. Inspect Results

- Review bootstrap confidence intervals (does CI exclude 0?)
- Check radial plots for architecture characteristics
- Analyze learning curves for qualitative differences

## Tools Reference

### run_icl_benchmark.sh

Launches 15 training runs with different seeds.

**Usage:**
```bash
./run_icl_benchmark.sh [OPTIONS]

Options:
  --architecture <arch>     vit_reset (default) or trxl
  --curriculum <style>      Curriculum style (default: terrain_2)
  --base-run-name <name>    Base name for runs (auto-generated if not specified)
  --skypilot                Launch on SkyPilot (default: local)
  --gpus <n>                Number of GPUs for SkyPilot (default: 4)
  --dry-run                 Print commands without executing
  --help                    Show this help message
```

**Examples:**
```bash
# Local training with vit_reset
./run_icl_benchmark.sh --architecture vit_reset --curriculum terrain_2

# SkyPilot training with trxl
./run_icl_benchmark.sh --architecture trxl --skypilot --gpus 8

# Dry run to preview commands
./run_icl_benchmark.sh --dry-run
```

### variance.py

Analyzes variance across multiple runs to determine minimum required sample size.

**Usage:**
```bash
python variance.py run_id_1 run_id_2 ... run_id_15 [OPTIONS]

Options:
  --metric <key>         Metric to analyze (default: overview/reward)
  --threshold <float>    Variance threshold as decimal (default: 0.05 = 5%)
  --output <path>        Output plot path (default: variance_analysis.png)
```

**Output:**
- Console: Reports minimum N where variance < threshold for AUC and Derivative
- Plot: Two-panel variance curves showing coefficient of variation vs sample size

**Example:**
```bash
python variance.py run1 run2 run3 run4 run5 run6 run7 run8 run9 run10 run11 run12 run13 run14 run15 \
    --metric overview/reward \
    --threshold 0.05 \
    --output icl_variance_analysis.png
```

### analysis.py

Statistical comparison of control vs treatment runs using bootstrap methods.

**Usage:**
```bash
# Paired comparison (same seeds for control and treatment)
python analysis.py \
    --pairs '[{"control": "run1", "candidate": "run2", "seed": 1072}, ...]' \
    --metric overview/reward

# Unpaired comparison (different seeds)
python analysis.py \
    --control-run-ids run1 run2 run3 \
    --candidate-run-ids run4 run5 run6 \
    --metric overview/reward
```

See `analysis.py` for full configuration options (summary specs, bootstrap settings, t-tests, power analysis).

## Best Practices

1. **Use paired seeds** when comparing architectures (reduces variance by 50-80%)
2. **Run variance analysis first** on pilot data to determine required N
3. **Match model capacity** between baseline and intervention (parameter count, FLOPS)
4. **Use same curriculum** for fair comparison
5. **Document everything**: git hash, hyperparameters, seed lists, W&B run IDs
6. **Check assumptions**: Review bootstrap distributions, test for normality if using t-tests
7. **Pre-register analysis plan**: Decide on metrics and thresholds before seeing results
