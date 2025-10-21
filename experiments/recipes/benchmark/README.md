# Architecture Benchmark Suite
My goal is to find conclusions like this:
"All our architectures fail on sparse+easy but succeed on sparse+hard → The problem isn't credit assignment, it's exploration"
"Architecture X dominates dense rewards but collapses on moderate → Over-reliance on reward shaping"
"Transformer scales better than LSTM but both fail on sparse → Need better exploration, not just capacity"
"This architecture has promising learning but the set of hyper parameters and the size and number of layers we've chosen is incorrect / insufficient"

A comprehensive 2-axis benchmark grid for testing agent architectures across **15 distinct conditions** (5 reward shaping levels × 3 task complexity levels).

## Complete Grid Structure

```
                    │  Easy      │ Medium     │  Hard      │
                    │ (1:1, init)│ (2:1, none)│ (3:1, none)│
────────────────────┼────────────┼────────────┼────────────┤
Dense Rewards       │     ✓      │     ✓      │     ✓      │
Moderate Rewards    │     ✓      │     ✓      │     ✓      │
Sparse Rewards      │     ✓      │     ✓      │     ✓      │
Terminal Only       │     ✓      │     ✓      │     ✓      │
Adaptive Curriculum │     ✓      │     ✓      │     ✓      │
```

**Standardized across all recipes:**
- Map size: 20×20
- Num agents: 20
- Combat: Enabled

**Total: 15 conditions × 13 architectures × 3 seeds = 585 runs**

## Design: Two Independent Axes

### 1. Task Complexity Axis (Easy/Medium/Hard)
**What makes the problem intrinsically harder:**

| Complexity | Converter Ratio | Initial Resources | Description |
|------------|----------------|-------------------|-------------|
| **Easy**   | 1:1 | Yes (2 in all buildings) | Simple resource chain, easier start |
| **Medium** | 2:1 | No | Moderate resource chain, standard start |
| **Hard**   | 3:1 (default) | No | Complex resource chain, standard start |

**Key insight:** Task complexity isolates how difficult the resource management problem is, independent of reward guidance.

### 2. Reward Shaping Axis (Dense/Moderate/Sparse/Terminal/Adaptive)
**How much guidance the agent gets:**

| Shaping | ore_red | battery_red | laser/armor | Description |
|---------|---------|-------------|-------------|-------------|
| **Dense** | 0.5 | 0.9 | 0.7 | High intermediate rewards - maximum guidance |
| **Moderate** | 0.2 | 0.5 | 0.3 | Medium intermediate rewards - balanced guidance |
| **Sparse** | 0.01 | 0.05 | 0.05 | Minimal intermediate rewards - limited guidance |
| **Terminal** | 0.0 | 0.0 | 0.0 | Only heart=1.0 - no intermediate guidance |
| **Adaptive** | Varies | Varies | Varies | Learning progress-guided curriculum |

**Note:** Converter ratios do NOT vary with reward shaping - they're fixed by task complexity.

## Scientific Benefits

This factorial design enables:
- **Isolate credit assignment**: Compare sparse vs dense on same task complexity
- **Isolate capacity/planning**: Compare easy vs hard on same reward shaping
- **Interaction effects**: "Does architecture X need dense rewards more on hard tasks?"
- **Statistical rigor**: Standard 2-factor ANOVA design

## Recipe Files

### Dense Rewards (0.5-0.9)
- `dense_easy.py` - High guidance, simple chain (1:1), initial resources
- `dense_medium.py` - High guidance, moderate chain (2:1)
- `dense_hard.py` - High guidance, complex chain (3:1)

### Moderate Rewards (0.2-0.5)
- `moderate_easy.py` - Balanced guidance, simple chain (1:1), initial resources
- `moderate_medium.py` - Balanced guidance, moderate chain (2:1)
- `moderate_hard.py` - Balanced guidance, complex chain (3:1)

### Sparse Rewards (0.01-0.05)
- `sparse_easy.py` - Limited guidance, simple chain (1:1), initial resources
- `sparse_medium.py` - Limited guidance, moderate chain (2:1)
- `sparse_hard.py` - Limited guidance, complex chain (3:1)

### Terminal Only (heart=1.0)
- `terminal_easy.py` - No guidance, simple chain (1:1), initial resources
- `terminal_medium.py` - No guidance, moderate chain (2:1)
- `terminal_hard.py` - No guidance, complex chain (3:1)

### Adaptive Curriculum
- `adaptive_easy.py` - Task variations around easy baseline
- `adaptive_medium.py` - Task variations around medium baseline
- `adaptive_hard.py` - Task variations around hard baseline

## Quick Start

### Train a Single Recipe

```bash
# Train with default architecture
uv run ./tools/run.py experiments.recipes.benchmark_architectures_and_losses.dense_easy.train

# Train with specific architecture
uv run ./tools/run.py experiments.recipes.benchmark_architectures_and_losses.sparse_hard.train \
  arch_type=vit_sliding

# Evaluate a trained policy
uv run ./tools/run.py experiments.recipes.benchmark_architectures_and_losses.dense_easy.evaluate \
  policy_uris=file://./checkpoints/my_policy
```

### Run Full Grid Sweep

```bash
# Run complete benchmark sweep with all architectures
cd experiments/recipes/benchmark_architectures_and_losses
uv run python -c "from benchmark import run; run('my_experiment', local=False)"

# Or customize the sweep
uv run python -c "
from benchmark import run, create_custom_grid
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

The adaptive curriculum recipes use learning progress-guided task selection with variations around each complexity baseline:

**adaptive_easy** (baseline: 20×20, 20 agents, 1:1 converter):
- Map size: 17-23×17-23
- Agents: 16-24
- Rewards: 0.1-0.7 (moderate range)
- Converter: 1:1 (fixed for easy)
- Initial resources: 0-1
- Pool: 32 tasks, 25% random

**adaptive_medium** (baseline: 20×20, 20 agents, 2:1 converter):
- Map size: 17-23×17-23
- Agents: 16-24
- Rewards: 0.05-0.6 (wider range)
- Converter: 2:1 (fixed for medium)
- Initial resources: 0-1
- Combat: on/off toggle
- Pool: 64 tasks, 25% random

**adaptive_hard** (baseline: 20×20, 20 agents, 3:1 converter):
- Map size: 17-28×17-28
- Agents: 16-28
- Rewards: 0.0-0.3 (terminal to sparse)
- Converter: 3:1 (fixed for hard)
- Initial resources: 0-1-2
- Combat: on/off toggle
- Pool: 128 tasks, 30% random

**Benefits:**
- Tests generalization across task variations
- More robust policies
- Better sample efficiency on diverse tasks
