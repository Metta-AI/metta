## LP Reward Time-Series Analysis – Implementation Plan

### Goals
1. Understand how the reward streams (the signals fed into the LP model) behave in *real* MettaGrid environments.
2. Quantify three aspects for each task/seed combination:
   - **Rate of change**: how quickly rewards trend upward/downward as training progresses.
   - **Sampling noise**: short-term variance of rewards across consecutive samples.
   - **Seed vs task variance**: how much two seeds of the *same* task differ compared to completely different tasks.
3. Turn those measurements into recommendations for the production LP configuration (EMA timescales, grouping, special handling of noisy tasks).

### Scope
- **Tasks**: focus on current curriculum rollouts:
  - `recipes.prod.cvc.fixed_maps.train` (with and without variants)
  - `recipes.experiment.assembly_lines.train`
  - One bleeding-edge recipe (e.g., `recipes.experiment.variants_curriculum.train` or `recipes.experiment.density_curriculum.train`)
  - For each, target ~3 seeds.
- **Data**: use existing training runs with full logging (WandB or stored rollouts). We need the ordered per-sample reward that LP consumes (`reward` from CurriculumEnv info).
- **Deliverables**: plots + summary document with clear guidance (e.g., “Task X is twice as noisy as Task Y – need longer EMA”).

### Data Extraction
1. **Identify runs**
   - Query WandB for recent runs of each target recipe (maybe use run tags).
2. **Download reward streams**
   - Use the WandB API (or existing log dumps) to pull `reward` vs `sample_index` for each `(task, seed)`.
   - Normalize by completion order so different seeds align (sample 0 = first completion in that task).
   - Store in `analysis/reward_timeseries/data/<task>/<seed>.parquet`.
3. **Metadata**
   - Record task label, seed, environment config, agent version, timestamp → `metadata.json`.

### Analysis Metrics
For each `(task, seed)` time series:
1. **Rate of change**
   - Fit an EMA (or rolling linear regression) over 25-sample windows to derive `d reward / d sample`.
   - Capture average slope + distribution per task.
2. **Sampling noise**
   - Rolling standard deviation (window=10 samples).
   - Plot noise vs sample index to see if noise shrinks as tasks get solved.
3. **Seed vs task variance**
   - Compute `Var(seed_mean_rewards within task)` vs `Var(task_means across tasks)`.
   - Ratio > 1 ⇒ seeds more different than tasks; ratio < 1 ⇒ tasks dominate variance.

### Visualizations
- Line plots: reward vs sample for each seed (colored per seed) within a task.
- Overlay of slopes/noise (shaded band) to highlight stability vs volatility.
- Box/violin plots comparing noise distributions across tasks.
- Heatmap or bar chart of variance ratios (seed vs task).

### Implementation Steps
1. **Data pull script**
   - `analysis/reward_timeseries/pull_wandb_data.py`
   - Inputs: project name, recipe list, seeds.
   - Outputs: parquet files + metadata.
2. **Analysis notebook/script**
   - `analysis/reward_timeseries/analyze_rewards.ipynb` (or `.py`)
   - Loads parquet files, computes metrics, generates plots.
3. **Summary doc**
   - `analysis/reward_timeseries/report.md`
   - Includes plots + actionable observations (e.g., “Chain crafting tasks show high intra-task variance – LP should widen EMA window”).

### Timeline (rough)
| Step | ETA |
| ---- | --- |
| Identify runs & write data pull script | 1 day |
| Download data + sanity check | 0.5 day |
| Implement metric calculations + plots | 1 day |
| Draft summary/report + review | 0.5 day |

### Open Questions / Assumptions
- Do we have enough runs with per-sample reward logged? (If not, may need to instrument upcoming training jobs.)
- Which recipes are highest priority for LP rollout? (Need confirmation to prioritize tasks.)
- Preferred format for final deliverable? (Slides vs markdown vs shared doc.)

### Next Actions
1. Confirm task list + runs to target.
2. Implement the data pull script and download an initial batch.
3. Prototype the metric notebook (rolling slope/variance, SPS correlation) using existing data so we can plug in new runs as they finish.
4. Share first plots + metrics for feedback before diving into deeper stats.

### Experiments Matrix

| Experiment | Description | Metrics to Monitor | Expected Outcome |
| --- | --- | --- | --- |
| Dual/annealed LP smoothing | Introduce fast + slow EMAs or temperature schedule for `z_score_amplification`. | Rolling reward slope/variance, convergence speed, SPS stability. | Reduced steady-state noise without sacrificing early progress. |
| GAE-based LP input | Replace `env_*` reward with episode-level GAE before normalization. | Same metrics as above + compare LP responsiveness. | More stable LP signal, especially post-convergence. |
| Active-pool caps / weighting | Limit active pool size or emphasize frontier tasks. | Sampling distribution (Gini/entropy), task throughput, SPS. | Higher sustained selectivity, reduced dilution, faster progression. |
| SPS-aware gating | Incorporate SPS or env step time into LP features/scoring. | Correlation between SPS dips and reward noise, overall efficiency. | LP deprioritizes slow/unstable envs, improving effective SPS. |

### Status Update (Dec 1)

- Metrics pipeline ready (`metrics_template.py`, `run_metrics.py`). Script auto-outputs rolling slope/std and SPS correlation plots once data is available.
- Experiments matrix drafted: dual smoothing, GAE input, active-pool controls, SPS-aware scoring (with monitoring metrics + expected outcomes).
- SkyPilot runs in flight: `prashant.fixed.maps.seed1`, `prashant.fixed.maps.seed2`, `prashant.variants.seed0`, `prashant.density.seed0` (all 8 GPUs). WandB report sections prepped for each run.
- Blocker: WandB full-history export still pending (documented attempts; plan to request server-side export if needed).
- Next steps: run metrics as soon as each job finishes, add plots to the WandB report, then test the LP tweaks defined above.

### Dual / Annealed Smoothing Plan

Goal: reduce steady-state reward noise without sacrificing early responsiveness.

**Config knobs:**
- `trainer.curriculum.algorithm_config.z_score_amplification` (current default ~10). Plan to sweep values like `{10, 20, 30}` combined with an annealing schedule (high early, lower later).
- Introduce a dual-timescale smoothing factor (if not exposed, add `trainer.curriculum.algorithm_config.slow_timescale_factor` variant or an override for `ema_timescale` split).

**Experiment setup:**
1. Baseline run (use existing seed0 data).
2. High-Z run: `z_score_amplification=30`.
3. Annealed run: start at 30 for first 10G steps, decay to 10 afterward (can approximate by scripting periodic overrides or supporting a config schedule).

**Metrics to monitor:**
- Rolling reward mean/std (`run_metrics.py`).
- Reward slope post-convergence.
- SPS correlation (ensure no regressions).
- WandB selectivity proxies (Gini/entropy if logged).

**Expected outputs:**
- Short write-up comparing reward noise before/after smoothing.
- Updated WandB plots/documented in report.

