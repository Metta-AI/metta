# Architecture Benchmark Suite

This directory contains 5 progressively difficult benchmark recipes for testing agent architectures, plus a script to automate testing across all combinations.

## Difficulty Levels

### Level 1 - Basic (`level_1_basic.py`)
**Easiest difficulty with maximum reward shaping**
- High intermediate rewards (0.5-0.9 for all resources)
- Easy 1:1 converter ratios
- Small map (15x15), fewer agents (12)
- Combat disabled (laser cost: 100)
- Initial resources in buildings
- **Use case**: Test basic learning capabilities and fast prototyping

### Level 2 - Easy (`level_2_easy.py`)
**Moderate reward shaping, still beginner-friendly**
- Moderate intermediate rewards (0.2-0.7)
- Standard 3:1 converter ratios
- Standard map (20x20), 16 agents
- Combat disabled
- No initial resources
- **Use case**: Test learning efficiency with less guidance

### Level 3 - Medium (`level_3_medium.py`)
**Combat enabled with reduced shaping**
- Low intermediate rewards (0.1-0.3)
- Standard converter ratios
- Standard map (25x25), 20 agents
- Combat enabled (laser cost: 1)
- Dual evaluation: basic and combat modes
- **Use case**: Test multi-task learning (resource gathering + combat)

### Level 4 - Hard (`level_4_hard.py`)
**Sparse rewards with full competition**
- Very sparse intermediate rewards (0.01-0.05)
- Full arena complexity, 24 agents
- Combat enabled
- Agents must discover strategies with minimal guidance
- **Use case**: Test exploration and credit assignment

### Level 5 - Expert (`level_5_expert.py`)
**Maximum difficulty with curriculum learning**
- Only heart reward (no intermediate rewards)
- Full combat, 24 agents
- Curriculum learning with task variations
- Agents must discover entire resource chain independently
- **Use case**: Test architecture's ability to learn complex behaviors from sparse feedback

## Quick Start

### Train on a Single Level

```bash
# Train with default ViT architecture
uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_1_basic.train

# Train with different architecture
uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_3_medium.train \
  policy_architecture=ViTSlidingTransConfig()

# Evaluate a trained policy
uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_1_basic.evaluate \
  policy_uri=file://./train_dir/my_run/checkpoints
```

### Run Full Benchmark Suite

The `run_benchmark.py` script tests 4 transformer variants across all 5 levels (20 total combinations):

```bash
# Run full benchmark (1M timesteps per run, ~20M total)
uv run experiments/recipes/benchmark_architectures/run_benchmark.py

# Run with fewer timesteps for quick testing
uv run experiments/recipes/benchmark_architectures/run_benchmark.py --max-timesteps 100000

# Test specific architectures
uv run experiments/recipes/benchmark_architectures/run_benchmark.py \
  --architectures vit,transformer

# Test specific levels
uv run experiments/recipes/benchmark_architectures/run_benchmark.py \
  --levels level_1_basic,level_3_medium

# Custom output directory
uv run experiments/recipes/benchmark_architectures/run_benchmark.py \
  --output-dir ./my_benchmark_results
```

## Architectures Tested

The benchmark script tests 4 architecture variants:

1. **ViT** (`ViTDefaultConfig`) - Default perceiver-based vision transformer
   - Perceiver attention over observation tokens
   - LSTM for temporal processing
   - Lightweight and fast

2. **ViT + Sliding Transformer** (`ViTSlidingTransConfig`) - ViT with transformer memory
   - Same perception as ViT
   - Sliding window transformer for memory
   - Better long-term dependencies

3. **Transformer** (`TransformerPolicyConfig`) - Standard transformer
   - Full transformer architecture
   - Positional encoding
   - More parameters, potentially higher capacity

4. **Fast** (`FastConfig`) - Fast LSTM baseline
   - Lightweight LSTM-based architecture
   - Minimal parameters
   - Baseline for speed/performance tradeoffs

## Benchmark Results

Results are saved to `./train_dir/benchmark/benchmark_results.json` with:
- Training duration and status
- Evaluation results for each run
- Command history for reproducibility

Example structure:
```json
{
  "start_time": "2025-01-15T10:30:00",
  "max_timesteps": 1000000,
  "runs": [
    {
      "run_number": 1,
      "total_runs": 20,
      "training": {
        "architecture": "vit",
        "level": "level_1_basic",
        "status": "success",
        "duration_seconds": 450.2
      },
      "evaluation": {
        "status": "success",
        "duration_seconds": 35.1
      }
    }
  ]
}
```

## Expected Difficulty Progression

As you move from Level 1 to Level 5, you should observe:

- **Sample efficiency**: Decreases (more samples needed to learn)
- **Final performance**: May plateau earlier at higher levels
- **Training stability**: Decreases (more variance)
- **Architecture differences**: Become more pronounced

Good architectures should:
- Learn quickly on Level 1-2
- Show stable learning on Level 3-4
- Demonstrate some learning progress on Level 5

## Usage Examples

### Quick Architecture Comparison

Compare two architectures on the easiest level:
```bash
uv run experiments/recipes/benchmark_architectures/run_benchmark.py \
  --architectures vit,fast \
  --levels level_1_basic \
  --max-timesteps 500000
```

### Full Scientific Benchmark

Run complete benchmark with sufficient training:
```bash
uv run experiments/recipes/benchmark_architectures/run_benchmark.py \
  --max-timesteps 5000000 \
  --output-dir ./results/full_benchmark_$(date +%Y%m%d)
```

### Test New Architecture

To add a new architecture to the benchmark:

1. Import it in `run_benchmark.py`:
   ```python
   from metta.agent.policies.my_arch import MyArchConfig
   ```

2. Add to `ARCHITECTURES` dict:
   ```python
   ARCHITECTURES = {
       "vit": ViTDefaultConfig(),
       "my_arch": MyArchConfig(),
       # ...
   }
   ```

3. Run benchmark:
   ```bash
   uv run experiments/recipes/benchmark_architectures/run_benchmark.py \
     --architectures my_arch
   ```

## Tips for Interpretation

1. **Level 1 performance**: Indicates basic learning capability and code correctness
2. **Level 1-3 progression**: Measures sensitivity to reward shaping
3. **Level 3-4 gap**: Tests exploration and credit assignment
4. **Level 5 performance**: Indicates ability to learn complex behaviors from sparse feedback
5. **Speed vs Performance**: Compare Fast baseline to understand parameter efficiency

## Common Issues

### Training Takes Too Long
- Reduce `--max-timesteps` for initial testing
- Start with `--levels level_1_basic` only
- Use `--architectures fast` for quick baseline

### Out of Memory
- The benchmark runs sequentially, so memory issues suggest a leak
- Check individual architecture memory requirements
- Consider reducing batch sizes in architecture configs

### Evaluation Fails
- Ensure training completed successfully (check train_dir)
- Verify checkpoint files exist at expected paths
- Use `--skip-training` to retry only evaluation

## Next Steps

After running the benchmark:

1. Analyze results in `benchmark_results.json`
2. Compare training curves in WandB (if enabled)
3. Identify which architectures excel at which difficulty levels
4. Use insights to guide architecture development
5. Consider hyperparameter tuning for promising architectures
