# Sweep Optimization Strategy

## How the Protein Optimizer Works

The Protein optimizer uses a **cost-aware Bayesian optimization** approach with the following key mechanisms:

### 1. Cost-Aware Acquisition Function

```python
target = (1 + expansion_rate) * np.random.rand()
weight = 1 - abs(target - gp_log_c_norm)
suggestion_scores = optimize_direction * max_c_mask * (gp_y_norm * weight)
```

- The `expansion_rate` controls exploration along the cost dimension
- Higher `expansion_rate` → more aggressive exploration of expensive configurations
- The optimizer balances performance (`gp_y_norm`) with cost preference (`weight`)

### 2. Progressive Cost Exploration

With `expansion_rate: 0.35`, the optimizer will:

- **Early runs (1-50)**: Prefer cheaper configurations (250M-500M timesteps)
- **Mid runs (50-150)**: Explore medium-cost configurations (500M-1.5B timesteps)
- **Late runs (150-250)**: Test expensive configurations (1.5B-3B timesteps)

This ensures you get:

- Quick initial feedback from cheap runs
- Progressive refinement with more expensive configurations
- Optimal use of compute budget

### 3. Key Parameter Settings

#### `num_random_samples: 15`

- First 15 runs are random exploration
- Builds initial GP model quickly
- Reduced from 20 for faster convergence

#### `expansion_rate: 0.35`

- Controls cost exploration aggressiveness
- 0.35 means target cost varies from 0 to 1.35x normalized cost range
- Higher than default 0.25 for more aggressive exploration

#### `resample_frequency: 10`

- Every 10th run, resample from Pareto frontier
- Ensures exploitation of known good configurations
- Prevents getting stuck in local optima

#### `suggestions_per_pareto: 128`

- Generate 128 candidates per Pareto point
- More candidates = better coverage of search space
- Balanced with computation time

#### `max_suggestion_cost: 7200`

- Maximum 2 hours per run
- Prevents runaway expensive configurations
- Can be adjusted based on compute budget

## Expected Behavior Over 250 Runs

### Phase 1: Random Exploration (Runs 1-15)

- Random sampling across parameter space
- Mix of cheap and expensive runs
- Building initial dataset for GP

### Phase 2: Early GP-Guided Search (Runs 16-50)

- GP model starts guiding search
- Focus on cheaper configurations
- Identifying promising parameter regions

### Phase 3: Cost Expansion (Runs 51-150)

- Progressive exploration of more expensive configurations
- Balancing exploration vs exploitation
- Building Pareto frontier of cost vs performance

### Phase 4: Refinement (Runs 151-250)

- Focus on high-performing configurations
- Testing expensive but promising settings
- Final optimization push

## Metric Choice: heart.get

Using `heart.get` instead of `reward` provides:

- **Direct signal**: Counts actual heart collections
- **Less noise**: Not averaged across 24 competing agents
- **Clearer optimization**: Directly optimizes the objective
- **Better convergence**: Stronger gradient for the optimizer

## Parameter Ranges Optimization

### Reduced Ranges for Faster Convergence

- **total_timesteps**: 250M-3B (was 500M-2B)
  - Lower min for cheaper initial exploration
  - Higher max for late-stage expensive runs

- **batch_size**: 2^17-2^20 (was 2^18-2^21)
  - Smaller batches for initial cheap runs
  - Controlled max to prevent OOM

- **update_epochs**: 1-4 (was 1-6)
  - Reduced max for faster iterations
  - Most gains come from 1-3 epochs

### Simplified Distributions

- Changed some `logit_normal` to `uniform` for simpler exploration
- Reduces GP modeling complexity
- Faster convergence with 250 samples

## Usage

```bash
# Launch sweep with improved configuration
./devops/skypilot/launch.py sweep \
    sweep=full_improved \
    sweep.metric=heart.get \
    sweep.protein.expansion_rate=0.35
```

## Monitoring Progress

Track these metrics to assess sweep health:

1. **Cost progression**: Should gradually increase over time
2. **Heart.get scores**: Should improve as optimizer learns
3. **Pareto frontier**: Should expand in both dimensions
4. **Exploration vs exploitation**: Check resample frequency hits

## Adjustments

If needed, you can tune:

- `expansion_rate`: Higher (0.4-0.5) for more aggressive cost exploration
- `resample_frequency`: Lower (5-7) for more exploitation
- `max_suggestion_cost`: Based on your compute budget
- `num_random_samples`: Lower (10) if you have good priors
