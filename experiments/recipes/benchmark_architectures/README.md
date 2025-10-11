# Architecture Benchmark Suite

This directory contains a comprehensive 2-axis benchmark grid for testing agent architectures across 12 distinct conditions (4 reward shaping levels × 3 task complexity levels), plus automated scripts to run the full suite.

## Design Note: Two Axes of Difficulty

**Status**: The benchmark now uses a 2-axis grid design that separates two independent variables for rigorous scientific comparison:

1. **Reward Shaping Axis** (holding task complexity constant)
   - Dense: High intermediate rewards for all resources (0.5-0.9)
   - Moderate: Medium intermediate rewards (0.2-0.5)
   - Sparse: Minimal intermediate rewards (0.01-0.1)
   - Terminal-only: Only final heart reward

2. **Task Complexity Axis** (holding reward structure constant)
   - Easy: Small map (15x15), few agents (12), no combat
   - Medium: Standard map (20x20), standard agents (16-20), optional combat
   - Hard: Large map (25x25), many agents (24), full combat

This 2-axis design would enable answering:
- How sensitive is architecture X to reward shaping?
- How does architecture X scale with task complexity?
- Are there interaction effects between the two?

### Complete Grid Structure (All Implemented)

|                    | Easy Map (15×15, 12 agents) | Medium Map (20×20, 20 agents) | Hard Map (25×25, 24 agents) |
|--------------------|----------------------------|-------------------------------|----------------------------|
| **Dense Rewards**  | Quick sanity check (✓)     | Standard benchmark (✓)        | Capacity test (✓)          |
| **Moderate Rewards** | Credit assignment (✓)    | Balanced difficulty (✓)       | Realistic challenge (✓)    |
| **Sparse Rewards** | Exploration baseline (✓)   | Reduced guidance (✓)          | Expert challenge (✓)       |
| **Terminal Only**  | Pure exploration (✓)       | Sparse + scaling (✓)          | Maximum difficulty (✓)     |

This provides **12 distinct conditions** for comprehensive architecture evaluation.

**Benefits for Scientific Publication**:
- Clear ablation: "Architecture X outperforms Y on sparse rewards across all task complexities"
- Falsifiable hypotheses: "Transformer architectures scale better than LSTMs to hard tasks"
- Interaction analysis: "Memory mechanisms help most when rewards are sparse AND tasks are complex"
- Standard factorial design enables ANOVA and other statistical analysis

## Current Difficulty Levels (Legacy)

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

The benchmark suite tests 13 architecture variants across all 12 grid cells with 3 random seeds each (468 total runs):

```bash
# Run full benchmark with adaptive controller (configured in the script)
uv run experiments/recipes/benchmark_architectures/adaptive.py
```

To customize the benchmark, edit the `if __name__ == "__main__"` section in `adaptive.py`:

```python
# Full grid sweep (all 12 cells)
run(
    experiment_id="benchmark_2axis_sweep",
    local=False,
    timesteps=2_000_000_000,
    max_parallel=16,
    seeds_per_cell=3,
    gpus=4,
    nodes=4,
)

# Test only reward shaping axis (holding complexity constant)
run(
    experiment_id="reward_shaping_sweep",
    grid=create_custom_grid(complexity_levels=["medium"]),
    timesteps=1_000_000,
    seeds_per_cell=3,
)

# Test only task complexity axis (holding rewards constant)
run(
    experiment_id="complexity_sweep",
    grid=create_custom_grid(reward_levels=["moderate"]),
    timesteps=1_000_000,
    seeds_per_cell=3,
)
```

The adaptive controller automatically:
- Runs all 13 architectures across all 12 grid cells with 3 random seeds (468 total runs)
- Manages training and evaluation jobs
- Tracks progress in WandB with 2-axis metadata
- Handles job scheduling and parallelization (default: 16 parallel jobs for 4 nodes × 4 GPUs)
- Supports both local and remote (Skypilot) execution
- Can be configured for selective grid cell testing

## Architectures Tested

The benchmark suite tests 13 architecture variants:

### Vision Transformer Variants
1. **ViT** (`ViTDefaultConfig`) - Default perceiver-based vision transformer
   - Perceiver attention over observation tokens
   - LSTM for temporal processing
   - Lightweight and fast

2. **ViT + Sliding Transformer** (`ViTSlidingTransConfig`) - ViT with transformer memory
   - Same perception as ViT
   - Sliding window transformer for memory
   - Better long-term dependencies

3. **ViT + Reset** (`ViTResetConfig`) - ViT with LSTM reset mechanism
   - ViT perception with reset-capable LSTM
   - Handles episode boundaries better
   - Improved temporal credit assignment

### Transformer Variants
4. **Transformer** (`TransformerPolicyConfig`) - Standard transformer
   - Full transformer architecture
   - Positional encoding
   - More parameters, potentially higher capacity

5. **GTrXL** (`gtrxl_policy_config()`) - Gated Transformer-XL
   - Gated Transformer-XL with GRU-style gating
   - Memory-based attention for long sequences
   - Better gradient flow

6. **TrXL** (`trxl_policy_config()`) - Transformer-XL
   - Transformer-XL with relative positional attention
   - Recurrence mechanism for long-term dependencies
   - Efficient memory management

7. **TrXL Nvidia** (`trxl_nvidia_policy_config()`) - NVIDIA-optimized Transformer-XL
   - NVIDIA-specific optimizations
   - Improved performance on GPU
   - Same architecture as TrXL

### Fast/LSTM Variants
8. **Fast** (`FastConfig`) - Fast LSTM baseline
   - Lightweight LSTM-based architecture
   - Minimal parameters
   - Baseline for speed/performance tradeoffs

9. **Fast LSTM Reset** (`FastLSTMResetConfig`) - Fast with LSTM reset
   - Fast architecture with reset mechanism
   - Better episode boundary handling
   - Improved credit assignment

10. **Fast Dynamics** (`FastDynamicsConfig`) - Fast with dynamics modeling
    - Fast architecture with auxiliary dynamics prediction
    - Learns world model alongside policy
    - Improved sample efficiency

### Specialized Architectures
11. **Memory Free** (`MemoryFreeConfig`) - Stateless ViT variant
    - No recurrent memory
    - Purely feedforward processing
    - Tests importance of temporal state

12. **AGaLiTe** (`AGaLiTeConfig`) - Attention-Gated Linear Transformer
    - Attention-Gated Linear Transformers
    - Efficient linear attention mechanism
    - Reduced computational complexity

13. **Puffer** (`PufferPolicyConfig`) - PufferLib baseline
    - CNN + LSTM architecture from PufferLib
    - Industry-standard baseline
    - Well-tested and optimized

## Benchmark Results

Results are tracked in WandB under the `metta-research/metta` project with:
- Training metrics and curves
- Evaluation results for each run
- Job status and metadata
- Pixels explored metric (displayed as `env_agent/pixels_explored`)

Each run is tagged with:
- `benchmark_architectures` - Main benchmark suite tag
- `<experiment_id>` - Your specified experiment identifier
- Architecture type, level, and seed metadata

Training checkpoints are saved to `./train_dir/<run_id>/checkpoints/` and evaluation results to `./train_dir/eval_<run_id>/stats.db`.

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

**Note**: Because reward shaping and task complexity are conflated in the current design, it's difficult to determine whether performance differences are due to:
- Sensitivity to sparse rewards (credit assignment, exploration)
- Inability to scale to complex tasks (capacity, memory)
- Both factors interacting

The proposed 2-axis design would disambiguate these effects.

## Usage Examples

### Quick Test Run

Test a single architecture on a single level:
```bash
# Train and evaluate on level 1 with fast architecture
uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_1_basic.train \
  run=quick_test \
  arch_type=fast \
  trainer.total_timesteps=500000

uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_1_basic.evaluate \
  policy_uri=file://./train_dir/quick_test/checkpoints
```

### Full Scientific Benchmark

Run complete 2-axis benchmark across all architectures and grid cells:
```bash
# Edit adaptive.py to configure experiment settings
uv run experiments/recipes/benchmark_architectures/adaptive.py
```

### Test New Architecture

To add a new architecture to the benchmark:

1. Import it in `level_1_basic.py`:
   ```python
   from metta.agent.policies.my_arch import MyArchConfig
   ```

2. Add to `ARCHITECTURES` dict in `level_1_basic.py`:
   ```python
   ARCHITECTURES = {
       "vit": ViTDefaultConfig(),
       "my_arch": MyArchConfig(),
       # ...
   }
   ```

3. The architecture will automatically be available in all level modules and `adaptive.py`

4. Run with adaptive controller:
   ```bash
   # Architecture will automatically be included in the grid sweep
   uv run experiments/recipes/benchmark_architectures/adaptive.py
   ```

## Tips for Interpretation

### Current (Legacy) Interpretation
1. **Level 1 performance**: Indicates basic learning capability and code correctness
2. **Level 1-3 progression**: Measures sensitivity to reward shaping (but also task complexity!)
3. **Level 3-4 gap**: Tests exploration and credit assignment (but also scaling!)
4. **Level 5 performance**: Indicates ability to learn complex behaviors from sparse feedback
5. **Speed vs Performance**: Compare Fast baseline to understand parameter efficiency

### Proposed 2-Axis Interpretation
With the grid design, you could isolate:

**Reward Shaping Sensitivity** (fixing task complexity):
- Compare Dense → Sparse along any column
- Architectures with good credit assignment show graceful degradation
- Large performance drops indicate need for dense rewards

**Task Scaling** (fixing reward structure):
- Compare Easy → Hard along any row
- Architectures with sufficient capacity maintain performance
- Degradation indicates capacity/memory limitations

**Interaction Effects**:
- Some architectures may excel at sparse rewards on easy tasks but fail on hard tasks
- Others may need dense rewards but scale well to complex environments

## Common Issues

### Training Takes Too Long
- Reduce `--timesteps` for initial testing
- Test individual levels using the recipe modules directly (e.g., `level_1_basic.train`)
- Use `arch_type=fast` for quick baseline tests

### Out of Memory
- Reduce `--max-parallel` to run fewer jobs concurrently
- Check individual architecture memory requirements
- Consider reducing batch sizes in architecture configs

### Evaluation Fails
- Ensure training completed successfully (check train_dir)
- Verify checkpoint files exist at expected paths
- Run evaluation manually using the recipe's evaluate function

## Next Steps

After running the benchmark:

1. Analyze results in `benchmark_results.json`
2. Compare training curves in WandB (if enabled)
3. Identify which architectures excel at which difficulty levels
4. Use insights to guide architecture development
5. Consider hyperparameter tuning for promising architectures
