# Architecture Benchmark Suite

A comprehensive 2-axis benchmark grid for testing agent architectures across **15 distinct conditions** (5 reward shaping levels × 3 task complexity levels).

## Complete Grid Structure

```
                    │  Easy Map  │ Medium Map │  Hard Map  │
                    │ (15×15,12) │ (20×20,20) │ (25×25,24) │
────────────────────┼────────────┼────────────┼────────────┤
Dense Rewards       │     ✓      │     ✓      │     ✓      │
Moderate Rewards    │     ✓      │     ✓      │     ✓      │
Sparse Rewards      │     ✓      │     ✓      │     ✓      │
Terminal Only       │     ✓      │     ✓      │     ✓      │
Adaptive Curriculum │     ✓      │     ✓      │     ✓      │
```

**Total: 15 conditions × 13 architectures × 3 seeds = 585 runs**

## Design: Two Independent Axes

### 1. Reward Shaping Axis (holding task complexity constant)
- **Dense**: High intermediate rewards (0.5-0.9) - maximum guidance
- **Moderate**: Medium intermediate rewards (0.2-0.7) - balanced guidance
- **Sparse**: Minimal intermediate rewards (0.01-0.1) - limited guidance
- **Terminal-only**: Only heart reward - no guidance
- **Adaptive**: Learning progress-guided curriculum with task variations

### 2. Task Complexity Axis (holding reward structure constant)
- **Easy**: Small map (15×15), 12 agents, no combat
- **Medium**: Standard map (20×20), 20 agents, optional combat
- **Hard**: Large map (25×25), 24 agents, full combat

## Scientific Benefits

This factorial design enables:
- **Clear ablations**: "Architecture X outperforms Y on sparse rewards across all complexities"
- **Falsifiable hypotheses**: "Transformers scale better than LSTMs to hard tasks"
- **Interaction analysis**: "Memory helps most when rewards are sparse AND tasks are complex"
- **Statistical rigor**: Standard factorial design for ANOVA and other analyses

## Recipe Files

### Dense Rewards
- `dense_easy.py` - Maximum guidance, minimal complexity
- `dense_medium.py` - High guidance, standard complexity
- `dense_hard.py` - High guidance, maximum complexity

### Moderate Rewards
- `moderate_easy.py` - Balanced guidance, minimal complexity
- `moderate_medium.py` - Balanced guidance, standard complexity
- `moderate_hard.py` - Balanced guidance, maximum complexity

### Sparse Rewards
- `sparse_easy.py` - Limited guidance, minimal complexity
- `sparse_medium.py` - Limited guidance, standard complexity
- `sparse_hard.py` - Limited guidance, maximum complexity

### Terminal Only
- `terminal_easy.py` - No guidance, minimal complexity
- `terminal_medium.py` - No guidance, standard complexity
- `terminal_hard.py` - No guidance, maximum complexity

### Adaptive Curriculum
- `adaptive_easy.py` - Task variations, minimal complexity
- `adaptive_medium.py` - Task variations, standard complexity
- `adaptive_hard.py` - Task variations, maximum complexity

## Quick Start

### Train a Single Recipe

```bash
# Train with default architecture
uv run ./tools/run.py experiments.recipes.benchmark_architectures.dense_easy.train

# Train with specific architecture
uv run ./tools/run.py experiments.recipes.benchmark_architectures.sparse_hard.train \
  arch_type=vit_sliding

# Evaluate a trained policy
uv run ./tools/run.py experiments.recipes.benchmark_architectures.dense_easy.evaluate \
  policy_uris=file://./checkpoints/my_policy
```

### Run Full Grid Sweep

```bash
# Run complete benchmark sweep with all architectures
cd experiments/recipes/benchmark_architectures
uv run python -c "from adaptive import run; run('my_experiment', local=False)"

# Or customize the sweep
uv run python -c "
from adaptive import run, create_custom_grid
grid = create_custom_grid(
    reward_levels=['dense', 'moderate'],
    complexity_levels=['easy', 'medium']
)
run('focused_experiment', grid=grid, architecture_types=['vit', 'fast'])
"
```

## Available Architectures

- `vit` - Vision Transformer (default)
- `vit_sliding` - ViT with sliding window attention
- `vit_reset` - ViT with reset mechanism
- `transformer` - Standard Transformer
- `fast` - Fast baseline policy
- `fast_lstm_reset` - Fast policy with LSTM reset
- `fast_dynamics` - Fast policy with dynamics model
- `memory_free` - Memory-free baseline
- `agalite` - AGaLiTe (Adaptive Gating and Lightweight Transformer)
- `gtrxl` - Gated Transformer-XL
- `trxl` - Transformer-XL
- `trxl_nvidia` - NVIDIA Transformer-XL variant
- `puffer` - Puffer policy

## Adaptive Curriculum Details

The adaptive curriculum recipes use learning progress-guided task selection:

**What varies:**
- Map size (±2-3 from baseline)
- Number of agents (±2-4 from baseline)
- Reward values (creating different credit assignment challenges)
- Combat on/off
- Initial resources in buildings

**Learning progress algorithm:**
- Maintains pool of 32-128 active tasks (depends on complexity)
- Scores tasks based on learning progress (EMA of performance)
- 25-30% random exploration rate
- Automatically focuses on tasks where agent is making progress

**Benefits:**
- Tests generalization across task variations
- More robust policies
- Better sample efficiency on diverse tasks
