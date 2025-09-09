# Sweep Validation Experiment Plan: Reproducibility and Fair Comparison in MARL HPO

## Executive Summary

This document outlines a comprehensive plan for validating high-performing hyperparameter configurations discovered through our 2B timestep sweep on `arena_basic_easy_shaped` using the `standard.ppo` configuration. The validation experiment aims to ensure fair comparison and reproducibility by re-running the best configurations with multiple seeds to establish statistical significance of performance improvements.

## Context and Purpose

### Current State
- Successfully completed large-scale hyperparameter sweep with 300 trials over 6 PPO parameters
- Identified several high-performing configurations achieving 25-50% performance improvements
- Need to validate that these improvements are robust and not artifacts of lucky random initialization

### Critical Challenge in MARL HPO
Based on comprehensive research, MARL hyperparameter optimization faces unique challenges:
- **Non-stationarity**: Multi-agent environments evolve as all agents learn simultaneously
- **Sparse rewards with long horizons**: Small hyperparameter changes can prevent reward discovery entirely
- **Insufficient seed sampling**: Most MARL studies use only 3-5 seeds despite recommendations for 10+ seeds
- **Extreme sensitivity**: Performance can vary drastically across seeds, especially in sparse reward settings

## Seed Control Analysis

### Q1: Is seed control part of the experimental pipeline?

**Yes**, seed control is implemented but with important caveats:

1. **Training Pipeline** (`metta/rl/system_config.py`):
   - `SystemConfig.seed` field with default random generation: `np.random.randint(0, 1000000)`
   - `seed_everything()` function sets seeds for:
     - Python's `random` module
     - NumPy's random state
     - PyTorch CPU and CUDA seeds
     - CuDNN deterministic mode (when `torch_deterministic=True`)
   - **Important Note**: Comment indicates "Despite these efforts, we still don't get deterministic behavior"
   - Distributed training adds rank-specific offsets to ensure different processes get different seeds

2. **Evaluation Pipeline**:
   - Uses the same `SystemConfig` infrastructure
   - Seeds are controlled through `system.seed` parameter
   - Each simulation run inherits seed settings from the system configuration

3. **Sweep Infrastructure**:
   - Currently **does NOT** explicitly manage seeds in `JobDefinition` or `RunInfo`
   - Seeds are implicitly randomized for each trial through default `SystemConfig` behavior

### Q2: How to specify seeds in training runs?

Seeds can be specified through the override mechanism:
```bash
./tools/run.py experiments.recipes.arena_basic_easy_shaped.train \
  --args run=validation_seed_42 \
  --overrides system.seed=42 system.torch_deterministic=true
```

### Q3: How to specify seeds in evaluations?

Similarly for evaluations:
```bash
./tools/run.py experiments.recipes.arena_basic_easy_shaped.evaluate_in_sweep \
  --args policy_uri=wandb://run/original_run \
  --overrides system.seed=42 system.torch_deterministic=true
```

### Q4: How many seeds for valid results?

Based on MARL HPO best practices:
- **Minimum**: 10 seeds (statistical significance threshold)
- **Recommended**: 15-20 seeds for high-confidence intervals
- **Ideal**: 30+ seeds for publication-quality results

For our validation experiment:
- **Phase 1**: 10 seeds per configuration (quick validation)
- **Phase 2**: 20 seeds for top 3 configurations (thorough validation)

## Required Modifications

### 1. Sweep Infrastructure Modifications

**Add seed tracking to `metta/sweep/models.py`:**
```python
@dataclass
class JobDefinition:
    # ... existing fields ...
    seed: Optional[int] = None  # Explicit seed for reproducibility
    
@dataclass
class RunInfo:
    # ... existing fields ...
    seed: Optional[int] = None  # Track seed used in run
```

### 2. New Validation Scheduler

Create `metta/sweep/schedulers/validation.py`:
```python
class ValidationScheduler:
    """Scheduler for reproducibility validation experiments."""
    
    def __init__(
        self,
        target_runs: list[str],  # WandB run IDs to validate
        seeds: list[int],  # Seeds to test each configuration
        recipe_module: str,
        train_entrypoint: str,
        eval_entrypoint: str,
    ):
        self.target_runs = target_runs
        self.seeds = seeds
        # ... initialization ...
    
    def schedule(self, sweep_metadata, all_runs, dispatched_trainings, dispatched_evals):
        """Generate validation jobs with controlled seeds."""
        jobs = []
        
        for target_run in self.target_runs:
            # Fetch original hyperparameters
            original_config = self._fetch_run_config(target_run)
            
            for seed in self.seeds:
                # Check if this seed/config combo already exists
                if not self._job_exists(target_run, seed, all_runs):
                    job = JobDefinition(
                        run_id=f"validation_{target_run}_seed_{seed}",
                        cmd=f"{self.recipe_module}.{self.train_entrypoint}",
                        overrides={
                            **original_config['observation']['suggestion'],
                            'system.seed': seed,
                            'system.torch_deterministic': 'true',
                            'wandb.group': f'validation_{target_run}',
                        },
                        seed=seed,
                        metadata={'original_run': target_run},
                    )
                    jobs.append(job)
        
        return jobs
```

### 3. Training Code Modifications

No modifications needed - existing seed control is sufficient when properly configured.

### 4. Evaluation Code Modifications

Ensure evaluations use consistent seeds by updating `SimTool` to accept and propagate seed:
```python
class SimTool(Tool):
    # ... existing fields ...
    seed: Optional[int] = None  # Explicit seed for evaluation
    
    def invoke(self, args, overrides):
        # ... existing code ...
        if self.seed is not None:
            self.system.seed = self.seed
        # ... rest of method ...
```

## Validation Experiment Design

### Phase 1: Quick Validation (10 seeds)
1. Select top 5 configurations from sweep
2. Run each with seeds: [42, 123, 456, 789, 1011, 1314, 1617, 1920, 2223, 2526]
3. Training: 2B timesteps (same as original)
4. Evaluation: 10 episodes per simulation
5. Total: 50 training runs + 50 evaluation runs

### Phase 2: Thorough Validation (20 seeds)
1. Select top 3 configurations from Phase 1
2. Add 10 more seeds: [3031, 3334, 3637, 3940, 4243, 4546, 4849, 5152, 5455, 5758]
3. Total: 30 additional training runs + 30 evaluation runs

### Success Metrics
- **Mean performance**: Must maintain ≥90% of original sweep performance
- **Standard deviation**: Should be <15% of mean
- **Confidence intervals**: 95% CI should not overlap with baseline

## Implementation Timeline

1. **Week 1**: Implement infrastructure changes
   - Add seed fields to models
   - Create ValidationScheduler
   - Test with single configuration

2. **Week 2**: Run Phase 1 validation
   - Launch 50 parallel training jobs
   - Monitor for failures
   - Collect initial statistics

3. **Week 3**: Analyze Phase 1 & Run Phase 2
   - Statistical analysis of Phase 1
   - Launch Phase 2 for top performers
   - Begin drafting results

4. **Week 4**: Final analysis and reporting
   - Complete statistical analysis
   - Generate visualization plots
   - Write final report

## Caveats and Considerations

### MARL-Specific HPO Challenges

1. **Non-determinism Despite Seed Control**: 
   - Multi-threading and GPU operations introduce unavoidable variance
   - Focus on statistical trends rather than exact reproducibility

2. **Computational Cost**:
   - Each validation run requires full 2B timesteps
   - Estimated: 500-1000 GPU-hours for complete validation

3. **Population Effects**:
   - Agent interactions create emergent behaviors sensitive to initialization
   - Small seed variations can lead to qualitatively different strategies

4. **Evaluation Consistency**:
   - Must use identical evaluation protocols for fair comparison
   - Consider both training convergence and final performance

## Recommendations

1. **Use Population-Based Validation**: Consider validating with different agent populations to test robustness

2. **Track Behavioral Metrics**: Beyond rewards, track coordination emergence, strategy diversity

3. **Implement Staged Validation**: Start with cheap proxies (shorter training) before full runs

4. **Create Validation Dashboard**: Real-time monitoring of validation progress and statistics

5. **Document Deviations**: Any run failures or anomalies must be carefully documented

## Conclusion

This validation experiment is essential for establishing the scientific validity of our hyperparameter optimization results. By implementing proper seed control and running systematic validation with 10-20 seeds per configuration, we can confidently claim that performance improvements are robust and not artifacts of lucky initialization. The infrastructure modifications are minimal but critical for enabling reproducible MARL research.

The extreme sensitivity of MARL systems to hyperparameters and seeds underscores the importance of this validation. Without it, we risk promoting configurations that work well only under specific random conditions, undermining the practical value of our optimization efforts.