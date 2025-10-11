# 2-Axis Grid Benchmark Complete

## Summary

The architecture benchmark suite now implements a complete 4×3 grid design, providing 12 distinct experimental conditions for rigorous scientific comparison.

## Complete Implementation

### Grid Overview

```
                   │  Easy Map  │ Medium Map │  Hard Map  │
                   │ (15×15,12) │ (20×20,20) │ (25×25,24) │
───────────────────┼────────────┼────────────┼────────────┤
Dense Rewards      │     ✓      │     ✓      │     ✓      │
Moderate Rewards   │     ✓      │     ✓      │     ✓      │
Sparse Rewards     │     ✓      │     ✓      │     ✓      │
Terminal Only      │     ✓      │     ✓      │     ✓      │
```

**Total: 12 conditions × 13 architectures × 3 seeds = 468 runs**

## Files Created

### New Recipe Modules (7 files)
1. ✅ `dense_medium.py` - Dense rewards on medium map
2. ✅ `dense_hard.py` - Dense rewards on hard map with combat
3. ✅ `moderate_easy.py` - Moderate rewards on easy map
4. ✅ `moderate_hard.py` - Moderate rewards on hard map with combat
5. ✅ `sparse_easy.py` - Sparse rewards on easy map
6. ✅ `terminal_easy.py` - Terminal-only rewards on easy map
7. ✅ `terminal_medium.py` - Terminal-only rewards on medium map

### Existing Recipe Modules (Mapped to Grid)
- `level_1_basic.py` → (dense, easy)
- `level_2_easy.py` → (moderate, medium)
- `level_3_medium.py` → (sparse, medium)
- `level_4_hard.py` → (sparse, hard)
- `level_5_expert.py` → (terminal_only, hard)

### Updated Files
- ✅ `adaptive.py` - Refactored to use 2-axis grid scheduler
- ✅ `README.md` - Updated with 2-axis design documentation
- ✅ `MIGRATION_2AXIS.md` - Migration guide and design rationale

## Reward Shaping Levels

### Dense Rewards (0.5-0.9)
- ore_red: 0.5
- battery_red: 0.9
- laser: 0.7
- armor: 0.7
- blueprint: 0.5
- 1:1 converter ratio
- Initial resources in buildings

### Moderate Rewards (0.2-0.7)
- ore_red: 0.2
- battery_red: 0.7
- laser: 0.4
- armor: 0.4
- blueprint: 0.3
- 3:1 converter ratio (standard)
- No initial resources

### Sparse Rewards (0.01-0.05)
- ore_red: 0.01
- battery_red: 0.05
- laser: 0.05
- armor: 0.05
- blueprint: 0.01
- 3:1 converter ratio (standard)
- No initial resources

### Terminal Only (0 intermediate)
- ore_red: 0
- battery_red: 0
- laser: 0
- armor: 0
- blueprint: 0
- heart: 1 (only terminal reward)
- 3:1 converter ratio (standard)
- No initial resources

## Task Complexity Levels

### Easy
- Map: 15×15
- Agents: 12
- Combat: Disabled (laser cost: 100)

### Medium
- Map: 20×20
- Agents: 20
- Combat: Disabled (laser cost: 100)

### Hard
- Map: 25×25
- Agents: 24
- Combat: Enabled (laser cost: 1)
- Dual evaluation: basic + combat modes

## Running the Benchmark

### Full Grid Sweep (All 468 Runs)
```bash
uv run experiments/recipes/benchmark_architectures/adaptive.py
```

### Test Reward Shaping Axis Only
Edit `adaptive.py`:
```python
run(
    experiment_id="reward_shaping_sweep",
    grid=create_custom_grid(complexity_levels=["medium"]),
    timesteps=1_000_000,
    seeds_per_cell=3,
)
```

### Test Task Complexity Axis Only
Edit `adaptive.py`:
```python
run(
    experiment_id="complexity_sweep",
    grid=create_custom_grid(reward_levels=["moderate"]),
    timesteps=1_000_000,
    seeds_per_cell=3,
)
```

### Test Specific Grid Cell
```bash
# Train on sparse rewards × easy map
uv run ./tools/run.py experiments.recipes.benchmark_architectures.sparse_easy.train

# Train on dense rewards × hard map
uv run ./tools/run.py experiments.recipes.benchmark_architectures.dense_hard.train
```

## Scientific Benefits

### Isolate Variables
- **Reward sensitivity**: Compare performance along reward axis (holding complexity constant)
- **Task scaling**: Compare performance along complexity axis (holding rewards constant)
- **Interaction effects**: Identify architectures that excel in specific combinations

### Factorial Design
- Standard 4×3 factorial design
- Enables ANOVA and interaction analysis
- Clear main effects for each axis
- Testable interaction hypotheses

### Publication-Ready Claims
✅ "Architecture X shows better credit assignment (reward axis)"
✅ "Architecture Y scales better to complex tasks (complexity axis)"
✅ "Memory mechanisms help most on sparse+hard (interaction effect)"

## Metadata Structure

Each run is tagged with:
- `benchmark/arch`: Architecture type (vit, fast, transformer, etc.)
- `benchmark/reward_shaping`: Reward level (dense, moderate, sparse, terminal_only)
- `benchmark/task_complexity`: Task level (easy, medium, hard)
- `benchmark/seed`: Random seed (00, 01, 02, ...)

Run ID format:
```
{experiment_id}.{arch}.{reward_shaping}.{task_complexity}.s{seed}

Example: benchmark_2axis.vit.sparse.hard.s00
```

## Next Steps

1. **Run Full Grid**: Execute the complete 468-run benchmark
2. **Analyze Results**: Use 2-axis structure for factorial analysis
3. **Identify Patterns**: Look for reward sensitivity vs. scaling patterns
4. **Test Hypotheses**: Evaluate interaction effects
5. **Publish Findings**: Use grid structure for clear scientific claims

## Code Quality

All files:
- ✅ Formatted with `ruff format`
- ✅ Linted with `ruff check --fix`
- ✅ Type hints included
- ✅ Docstrings with grid position
- ✅ Consistent structure across all recipes

