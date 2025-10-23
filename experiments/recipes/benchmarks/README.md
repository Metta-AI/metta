## Available Benchmarks

### 1. arena_prog_7
Multi-agent arena environment with resource collection and conversion.
- 24 agents
- 70x70 map
- Hearts as primary reward (max 100)
- Attacks disabled (laser cost = 100)

### 2. navigation
Single-agent navigation tasks across varied terrain.
- 1 agent per instance, 4 instances
- Varied terrain types (dense, sparse, balanced, maze, cylinder-world)
- Multiple map sizes (small, medium, large)

### 3. assembly_lines
In-context learning task with resource chains.
- 1 agent
- Variable chain lengths and sink configurations
- Level 0 difficulty for benchmarking

## Benchmark Matrix

| Architecture | Hyperparameters | Seeds | Map |

## Benchmarking Process

### Step 1: Run Hyperparameter Sweep

For each architecture and map combination, run a hyperparameter sweep to identify optimal configurations.

```bash
# Example: ViT_PPO on arena_prog_7
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.sweep \
  sweep_name="vit_ppo.arena.sweep" \
  -- gpus=4 nodes=2

# Example: Navigation sweep
uv run ./tools/run.py experiments.recipes.benchmarks.navigation.sweep \
  sweep_name="vit_ppo.navigation.sweep" \
  -- gpus=4 nodes=2

# Example: Assembly lines sweep
uv run ./tools/run.py experiments.recipes.benchmarks.assembly_lines.sweep \
  sweep_name="vit_ppo.assembly.sweep" \
  -- gpus=4 nodes=2
```

**Local testing before remote sweep:**
```bash
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.sweep \
  sweep_name="vit_ppo.arena.local_test" \
  -- local_test=True
```

### Step 2: Select Top Hyperparameters

After sweep completes:
1. Analyze sweep results in WandB
2. Select top 2-5 hyperparameter configurations based on:
   - Final score on `evaluator/eval_sweep/score` metric
   - Learning stability (low variance)
   - Sample efficiency (faster convergence)
3. Record selected hyperparameters as h1, h2, etc. in the benchmark matrix

### Step 3: Run Full Training Runs

For each selected hyperparameter configuration, run multiple training runs with different seeds.

```bash
# Example: Full training run with specific hyperparameters
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.train \
  run="vit_ppo.arena.h1.seed1" \
  trainer.learning_rate=3e-4 \
  trainer.ppo_clip_coef=0.2 \
  trainer.ppo_gae_lambda=0.95 \
  trainer.ppo_vf_coef=0.5 \
  trainer.adam_eps=1e-5 \
  trainer.total_timesteps=1e9
```

**Recommended number of seeds:** 3-5 per configuration

### Step 4: Evaluate Policies

Run comprehensive evaluation on trained policies:

```bash
# Example: Evaluate a trained policy
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.evaluate \
  policy_uris="file://./train_dir/vit_ppo.arena.h1.seed1/checkpoints"

# Or from S3/WandB
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.evaluate \
  policy_uris="s3://bucket/path/to/checkpoints/run:v100.pt"
```

### Step 5: Analyze Results

Generate comparative analysis across architectures, hyperparameters, and seeds:

```bash
# Example: Analyze evaluation database
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.analyze \
  eval_db_uri=./train_dir/eval_arena/stats.db
```

Key metrics to compare:
- **Final score**: Average reward across evaluation episodes
- **Sample efficiency**: Timesteps to reach target performance
- **Stability**: Variance across seeds
- **Generalization**: Performance on held-out evaluation tasks

## Recipe-Specific Details

### arena_prog_7

**Training environment:**
- Fixed 70x70 arena
- 24 agents
- Single reward: hearts (max 100)
- Altar converter: 1 battery_red → 1 heart

**Evaluation:**
- Same configuration as training (no curriculum variation)

**Commands:**
```bash
# Train
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.train run="my_run"

# Evaluate
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.evaluate policy_uris="..."

# Sweep
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.sweep sweep_name="my_sweep"
```

### navigation

**Training environment:**
- Fixed 4-instance map configuration
- 1 agent per instance
- Terrain: varied_terrain/dense_large
- 10 altars per instance

**Evaluation:**
- Multiple terrain types and map sizes
- Tests generalization to unseen terrains

**Commands:**
```bash
# Train
uv run ./tools/run.py experiments.recipes.benchmarks.navigation.train run="my_run"

# Evaluate
uv run ./tools/run.py experiments.recipes.benchmarks.navigation.evaluate policy_uris="..."

# Sweep
uv run ./tools/run.py experiments.recipes.benchmarks.navigation.sweep sweep_name="my_sweep"
```

### assembly_lines

**Training environment:**
- Fixed level_0 configuration
- 1 agent
- Chain length: 1
- Sinks: 0-1
- Room size: tiny
- No terrain

**Evaluation:**
- Tests on harder configurations (2-5 chain length, larger rooms, with terrain)
- Evaluates in-context learning capability

**Commands:**
```bash
# Train
uv run ./tools/run.py experiments.recipes.benchmarks.assembly_lines.train run="my_run"

# Evaluate
uv run ./tools/run.py experiments.recipes.benchmarks.assembly_lines.evaluate policy_uris="..."

# Sweep
uv run ./tools/run.py experiments.recipes.benchmarks.assembly_lines.sweep sweep_name="my_sweep"
```

## Enabling Curriculum Learning

By default, all benchmarks use **fixed environments** for fair comparison. To enable curriculum learning:

```bash
# Enable curriculum for any recipe
uv run ./tools/run.py experiments.recipes.benchmarks.arena_prog_7.train \
  run="with_curriculum" \
  use_curriculum=True
```

This is useful for comparing architectures with and without curriculum learning (e.g., ViT_PPO vs ViT_PPO_CL).

## Results Tracking

Track results in a shared spreadsheet or database:
- Record run name, architecture, hyperparameters, seed
- Log final evaluation scores
- Track training time and compute resources
- Document any anomalies or special observations

## Notes

- All recipes use ViT architecture by default (can be overridden with `policy_architecture` parameter)
- Sweeps use dedicated "sweep" evaluation suite to avoid metric namespace conflicts
- Combat mode is permanently disabled in arena_prog_7 for consistency
- The `__init__.py` file is required for Python to import these recipes as a package
