# Bayesian Optimization Parameter Selection Rationale

## The Curse of Dimensionality in BO

Bayesian Optimization becomes ineffective beyond ~10-15 parameters due to:

1. **Exponential search space growth**: Each parameter multiplies the volume
2. **GP model degradation**: Gaussian Processes scale O(n³) and struggle to model high-dim spaces
3. **Sparse data coverage**: Limited budget means poor coverage of high-dim spaces
4. **Acquisition function inefficiency**: Harder to balance exploration/exploitation

## Parameter Importance Ranking for PPO

Based on empirical studies and sensitivity analysis, PPO parameters can be ranked by impact:

### Tier 1: Critical Parameters (MUST OPTIMIZE)

1. **total_timesteps** - Determines convergence and final performance
2. **learning_rate** - Most sensitive parameter, wrong value = training failure
3. **batch_size** - Affects sample efficiency and stability
4. **gamma** - Discount factor fundamentally changes the objective
5. **gae_lambda** - Controls bias-variance tradeoff in advantage estimation

### Tier 2: High Impact (SHOULD OPTIMIZE)

6. **ent_coef** - Entropy bonus crucial for exploration
7. **clip_coef** - Core PPO parameter, but robust to small changes
8. **minibatch_size** - Affects gradient noise and convergence speed

### Tier 3: Moderate Impact (OPTIONAL)

9. **vf_coef** - Value function coefficient, default 0.5 usually works
10. **update_epochs** - More epochs can help but increases compute
11. **bptt_horizon** - Important for recurrent policies, less so otherwise

### Tier 4: Low Impact (FIX AT DEFAULTS)

- **vf_clip_coef** - Rarely needs tuning from 10.0
- **optimizer.beta1/beta2** - Adam defaults (0.9, 0.999) are robust
- **optimizer.eps** - Negligible impact, 1e-8 is fine

## Recommended Configuration

### For 8-parameter budget (Optimal for BO):

1. total_timesteps
2. learning_rate
3. batch_size
4. gamma
5. gae_lambda
6. ent_coef
7. clip_coef
8. minibatch_size

### For 10-parameter budget (Maximum recommended):

Add: 9. vf_coef 10. update_epochs

### Fixed at Good Defaults:

- bptt_horizon: 16 (or 64 for complex sequential tasks)
- vf_coef: 0.5 (if not optimizing)
- vf_clip_coef: 10.0
- update_epochs: 3 (if not optimizing)
- optimizer.beta1: 0.9
- optimizer.beta2: 0.999
- optimizer.eps: 1e-8

## Why These Choices?

### Keep:

- **total_timesteps**: Directly controls compute budget and convergence
- **learning_rate**: Single most impactful parameter on training dynamics
- **batch_size**: Affects both performance and computational efficiency
- **gamma/gae_lambda**: Fundamental to credit assignment and advantage estimation
- **ent_coef**: Controls exploration, critical for avoiding local optima
- **clip_coef**: Defines the trust region size, core to PPO

### Drop:

- **Adam betas/eps**: Extremely robust defaults, minimal gains from tuning
- **vf_clip_coef**: Almost never needs adjustment
- **bptt_horizon**: More architecture-dependent than algorithm-dependent

## Implementation Strategy

1. **Start with 8 parameters** for efficient BO convergence
2. **Use domain knowledge** to set tight bounds on each parameter
3. **Consider parameter interactions**:
   - batch_size and minibatch_size should maintain reasonable ratio
   - learning_rate may need to scale with batch_size
4. **Monitor effective dimensionality**: Some parameters may converge quickly and can be fixed

## Alternative Approaches for High-Dim Spaces

If you must optimize 15+ parameters:

1. **Random Search**: Often beats BO in high dimensions
2. **Population-based Training (PBT)**: Dynamically adjusts parameters during training
3. **Successive Halving**: Quickly eliminate bad configurations
4. **BOHB**: Combines BO with Hyperband for better scaling
