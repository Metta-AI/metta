# Migration to 2-Axis Grid Design

## Summary

The benchmark architecture suite has been refactored from a linear 5-level progression to a scientifically rigorous 2-axis grid design.

## Key Changes

### Before (Linear Progression)
- 5 ambiguous "levels" that conflated two variables
- Difficult to isolate what caused performance differences
- Limited scientific rigor

### After (2-Axis Grid)
- **Reward Shaping Axis** (vertical): dense → moderate → sparse → terminal_only
- **Task Complexity Axis** (horizontal): easy → medium → hard
- Clear separation of concerns
- Enables factorial analysis

## Complete Grid Status

```
                   │  Easy Map  │ Medium Map │  Hard Map  │
                   │ (15×15,12) │ (20×20,20) │ (25×25,24) │
───────────────────┼────────────┼────────────┼────────────┤
Dense Rewards      │     ✓      │     ✓      │     ✓      │
Moderate Rewards   │     ✓      │     ✓      │     ✓      │
Sparse Rewards     │     ✓      │     ✓      │     ✓      │
Terminal Only      │     ✓      │     ✓      │     ✓      │
```

✓ = All 12 grid cells are now implemented and enabled

## Usage Examples

### Full Grid Sweep (All Enabled Cells)
```python
uv run experiments/recipes/benchmark_architectures/adaptive.py
```

### Test Only Reward Shaping Axis
```python
from experiments.recipes.benchmark_architectures.adaptive import run, create_custom_grid

run(
    experiment_id="reward_shaping_sweep",
    grid=create_custom_grid(complexity_levels=["medium"]),
    timesteps=1_000_000,
    seeds_per_cell=3,
)
```

### Test Only Task Complexity Axis
```python
from experiments.recipes.benchmark_architectures.adaptive import run, create_custom_grid

run(
    experiment_id="complexity_sweep",
    grid=create_custom_grid(reward_levels=["moderate"]),
    timesteps=1_000_000,
    seeds_per_cell=3,
)
```

### Test Specific Architectures
```python
from experiments.recipes.benchmark_architectures.adaptive import run

run(
    experiment_id="vit_variants_sweep",
    architecture_types=["vit", "vit_sliding", "vit_reset"],
    timesteps=1_000_000,
    seeds_per_cell=3,
)
```

## Metadata Changes

### Old Run ID Format
```
{experiment_id}.{arch_type}.{level}.s{seed:02d}
Example: benchmark.vit.level_1_basic.s00
```

### New Run ID Format
```
{experiment_id}.{arch_type}.{reward_shaping}.{task_complexity}.s{seed:02d}
Example: benchmark.vit.dense.easy.s00
```

### Old Metadata
- `benchmark/arch`: Architecture type
- `benchmark/seed`: Random seed
- `benchmark/level`: Level name (ambiguous)

### New Metadata
- `benchmark/arch`: Architecture type
- `benchmark/seed`: Random seed
- `benchmark/reward_shaping`: Reward shaping level (dense/moderate/sparse/terminal_only)
- `benchmark/task_complexity`: Task complexity level (easy/medium/hard)

## Scientific Benefits

### Isolate Variables
- **Reward shaping sensitivity**: Compare performance across reward axis (holding complexity constant)
- **Task scaling**: Compare performance across complexity axis (holding rewards constant)
- **Interaction effects**: Identify architectures that excel in specific (reward, complexity) combinations

### Statistical Analysis
- Factorial design enables ANOVA
- Clear main effects and interaction effects
- Proper control conditions

### Publication-Ready Claims
- "Architecture X outperforms Y on sparse rewards across all task complexities" ✓
- "Transformer architectures scale better than LSTMs to complex tasks" ✓
- "Memory mechanisms help most when rewards are sparse AND tasks are complex" ✓

## Recipe Module Mapping

The complete 4×3 grid maps to these recipe files:

**Dense Rewards:**
- `level_1_basic.py` → (dense, easy)
- `dense_medium.py` → (dense, medium)
- `dense_hard.py` → (dense, hard)

**Moderate Rewards:**
- `moderate_easy.py` → (moderate, easy)
- `level_2_easy.py` → (moderate, medium)
- `moderate_hard.py` → (moderate, hard)

**Sparse Rewards:**
- `sparse_easy.py` → (sparse, easy)
- `level_3_medium.py` → (sparse, medium)
- `level_4_hard.py` → (sparse, hard)

**Terminal Only:**
- `terminal_easy.py` → (terminal_only, easy)
- `terminal_medium.py` → (terminal_only, medium)
- `level_5_expert.py` → (terminal_only, hard)

All recipe modules follow the same pattern, varying:
- `arena_env.game.agent.rewards.inventory` for reward shaping
- `arena_env.game.map_builder.width/height` for map size
- `num_agents` for agent count
- `arena_env.game.actions.attack.consumed_resources["laser"]` for combat

## Backwards Compatibility

Old code using level modules directly still works:
```bash
uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_1_basic.train
```

The adaptive controller has been updated to use the new grid structure, but individual recipe modules are unchanged.

