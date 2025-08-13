# Phased Sweep Strategy

## Why Manual Phasing?

The Protein optimizer's `expansion_rate` doesn't create progressive cost exploration over time. It just adds random
variation to cost preference at each step. Manual phasing gives us explicit control over the
exploration-exploitation-cost tradeoff.

## Three Phase Approach

### Phase 1: Broad Exploration with Cheap Runs (80 runs)

**Goal**: Map the hyperparameter landscape quickly

- **Timesteps**: Full range (100M-5B), biased toward 500M
- **Cost limit**: 30 minutes max (naturally limits to ~100M-500M)
- **Ranges**: WIDE - exploring full parameter space
- **Random samples**: 30 (high exploration)
- **Resampling**: Disabled (pure exploration)

**Key decisions**:

- Wide ranges to discover unexpected good regions
- Many random samples to build good initial GP model
- Cost control keeps runs cheap despite wide timestep range

### Phase 2: Exploitation with Medium Cost (80 runs)

**Goal**: Refine promising regions with better signal

- **Timesteps**: Same range (100M-5B), biased toward 1.5B
- **Cost limit**: 1 hour (naturally selects ~500M-1.5B)
- **Ranges**: SAME as Phase 1 (avoids normalization issues!)
- **Random samples**: 5 (mostly guided)
- **Resampling**: Every 5 runs (balanced)

**Key decisions**:

- CRITICAL: Same parameter ranges prevent observation corruption
- Higher cost limit for more reliable signal
- Frequent resampling from Pareto frontier
- Loading Phase 1 results as prior

### Phase 3: Final Push with Expensive Runs (90 runs)

**Goal**: Achieve peak performance

- **Timesteps**: Same range (100M-5B), biased toward 3B
- **Cost limit**: 3 hours (naturally selects ~1.5B-5B)
- **Ranges**: SAME as previous phases
- **Random samples**: 0 (pure exploitation)
- **Resampling**: Every 3 runs (heavy exploitation)

**Key decisions**:

- Still keeping reasonable range width for robustness
- Very high timesteps for best performance
- Aggressive exploitation of best configurations
- Loading all previous results

## Range Progression Philosophy

### CRITICAL INSIGHT: Parameter Range Consistency

**The Bug We Fixed**: When phases have different parameter ranges, observations from earlier phases get **corrupted**
when loaded into later phases. A 300M timestep value from Phase 1 (middle of 100M-500M range) gets clamped to 500M when
loaded into Phase 2 (which expected 500M-1.5B range), destroying the learned knowledge!

### What We AVOID:

- **Different ranges across phases**: This corrupts observations!
- **Over-narrowing**: Ranges that are too tight can miss better nearby optima
- **Premature convergence**: Getting stuck in local optima from Phase 1
- **Over-reliance on cheap runs**: Phase 1 results may not generalize

### What We DO:

- **Consistent ranges**: ALL phases use the SAME parameter ranges
- **Cost control**: Use `max_suggestion_cost` to control run expense
- **Mean biasing**: Different `mean` values guide exploration
- **Uniform distributions**: Simpler than logit_normal for most parameters

## Parameter-Specific Strategy

### High Confidence (can narrow more):

- **gamma**: Usually optimal around 0.99
- **beta2**: Rarely needs tuning from 0.999
- **eps**: Standard Adam default works well

### Medium Confidence (moderate narrowing):

- **learning_rate**: Important but problem-dependent
- **clip_coef**: PPO-specific, moderate range needed
- **gae_lambda**: Important for credit assignment

### Low Confidence (keep wide):

- **vf_coef**: Very problem-dependent
- **ent_coef**: Exploration needs vary greatly
- **vf_clip_coef**: Less studied, keep exploring

## Usage

```bash
# Phase 1: Exploration
./devops/skypilot/launch.py sweep \
    sweep=phase1_explore_cheap \
    sweep.metric=heart.get \
    wandb.name=my_sweep_phase1

# After ~80 runs, move to Phase 2
./devops/skypilot/launch.py sweep \
    sweep=phase2_exploit_medium \
    sweep.metric=heart.get \
    sweep.max_observations_to_load=80 \
    wandb.name=my_sweep_phase2

# After ~160 total runs, move to Phase 3
./devops/skypilot/launch.py sweep \
    sweep=phase3_final_expensive \
    sweep.metric=heart.get \
    sweep.max_observations_to_load=160 \
    wandb.name=my_sweep_phase3
```

## Expected Outcomes

### Phase 1 (runs 1-80):

- High variance in scores
- Quick identification of bad regions
- Discovery of 2-3 promising parameter clusters
- Total cost: ~40 GPU-hours

### Phase 2 (runs 81-160):

- Convergence toward better scores
- Clearer Pareto frontier emerging
- More consistent performance
- Total cost: ~80 GPU-hours

### Phase 3 (runs 161-250):

- Best overall scores
- Refined optimal configurations
- High confidence in results
- Total cost: ~200 GPU-hours

## Total Compute Budget

- Phase 1: ~40 GPU-hours
- Phase 2: ~80 GPU-hours
- Phase 3: ~200 GPU-hours
- **Total: ~320 GPU-hours**

This is more efficient than 250 runs at uniform high cost (would be ~500+ GPU-hours).
