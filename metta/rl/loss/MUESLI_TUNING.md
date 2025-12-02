# Muesli Loss Coefficient Tuning Guide

## Analysis from Run: local.daveey.20251202.153033

### Loss Magnitudes Observed

| Loss Component | Mean Value | Scale | Status |
|----------------|------------|-------|--------|
| **muesli_policy_loss** | **5.055** | ~5.0 | ⚠️ TOO HIGH |
| dynamics_returns_loss | 0.089 | ~0.1 | ✓ Good |
| dynamics_reward_loss | 0.012 | ~0.01 | ✓ Good |
| PPO policy_loss | -0.001 | ~0.001 | ✓ Good |
| PPO value_loss | 0.0002 | ~0.0002 | ✓ Good |
| entropy | 5.023 | ~5.0 | Note: high |

### Problem Identified

**Muesli policy loss was ~1000x larger than PPO losses**, causing it to dominate training. This happens because:

1. **Cross-entropy loss scale**: When predicting a nearly-uniform policy distribution, cross-entropy ≈ log(num_actions) ≈ 5.0 for typical action spaces
2. **PPO losses are already scaled**: PPO policy/value losses are naturally small (< 0.01)
3. **No coefficient balancing**: Default coefficient of 1.0 was too high

### Solutions Applied

#### 1. Reduced Muesli Policy Coefficient ✓

Changed in [`muesli.py`](muesli.py:40):
```python
# Before
policy_pred_coef: float = Field(default=1.0, ge=0.0)

# After
policy_pred_coef: float = Field(default=0.001, ge=0.0)
```

**Effect**: Brings Muesli loss contribution to ~0.005, comparable to other losses.

#### 2. Enable CMPO (Recommended)

CMPO was **NOT enabled** in this run. To enable:

```python
from recipes.experiment.cogs_v_clips import train

tt = train(num_cogs=4, mission="easy_hearts")

# Enable CMPO for better target policies
tt.trainer.losses.cmpo.enabled = True
tt.trainer.losses.cmpo.kl_coef = 0.01  # Start small

# Enable Muesli with adjusted coefficient
tt.trainer.losses.muesli_model.enabled = True
# (already using default 0.001 coefficient)
```

Or via CLI:
```bash
./devops/run.sh recipes.experiment.cogs_v_clips.train \
    trainer.losses.cmpo.enabled=true \
    trainer.losses.cmpo.kl_coef=0.01 \
    trainer.losses.muesli_model.enabled=true
```

## Recommended Coefficient Values

### Conservative (Start Here)

```python
# Muesli
muesli_model:
    enabled: true
    policy_pred_coef: 0.001  # ✓ Already updated
    policy_horizon: 1

# CMPO
cmpo:
    enabled: true
    kl_coef: 0.01  # Small to start

# Dynamics (if using unrolling)
dynamics:
    enabled: true
    returns_pred_coef: 1.0
    reward_pred_coef: 1.0
    unroll_steps: 3  # Start with fewer steps
```

**Expected loss magnitudes**:
- Muesli: ~0.005 (5.0 * 0.001)
- CMPO: ~0.01-0.05 (depends on advantage distribution)
- Dynamics: ~0.1 total
- PPO: ~0.001-0.01

### Aggressive (After Validation)

Once training is stable, you can increase:

```python
muesli_model:
    policy_pred_coef: 0.005  # 5x increase
    policy_horizon: 5  # Predict further ahead

cmpo:
    kl_coef: 0.05  # Stronger regularization

dynamics:
    unroll_steps: 5  # More unrolling
```

## Loss Ordering Matters

Losses execute in this order (from [`losses.py`](losses.py)):
1. `ppo_critic` - Value function
2. **`cmpo`** - Compute target policies ← Must run before Muesli!
3. `ppo_actor` - Policy gradient
4. **`dynamics`** - Unroll and predict values/rewards ← Must run before Muesli!
5. **`muesli_model`** - Policy prediction using unrolled logits

This ordering ensures:
- CMPO policies are available for Muesli targets
- Dynamics unrolling provides predictions for Muesli

## Monitoring Guidelines

### Key Metrics to Watch

1. **Loss Ratios**:
   ```python
   muesli_loss / ppo_policy_loss  # Should be ~1-10x, not 1000x
   ```

2. **Entropy**:
   - High entropy (>4.0): Policy is too random, Muesli might be overpowering
   - Low entropy (<1.0): Policy is too deterministic, might need more exploration
   - Target: ~2-3 for good exploration/exploitation

3. **Explained Variance**:
   - Should increase over training (target: >0.5)
   - Negative values indicate poor value predictions

4. **CMPO Metrics** (when enabled):
   - `cmpo_kl_loss`: Should be small (~0.01-0.1)
   - `cmpo_adv_mean`: Shows advantage distribution
   - `cmpo_adv_std`: Should be reasonable (not too small)

### Red Flags

⚠️ **Stop and reduce coefficients if you see**:
- Any single loss >10x larger than others
- Entropy collapsing to near-zero too quickly
- Explained variance staying negative
- Training becoming unstable (loss spikes)

## Testing Protocol

1. **Baseline** (no Muesli):
   ```bash
   # Run without Muesli to get baseline
   ./devops/run.sh recipes.experiment.cogs_v_clips.train \
       trainer.total_timesteps=10000
   ```

2. **Muesli only** (current coefficients):
   ```bash
   ./devops/run.sh recipes.experiment.cogs_v_clips.train \
       trainer.losses.muesli_model.enabled=true \
       trainer.total_timesteps=10000
   ```

3. **Muesli + CMPO** (recommended):
   ```bash
   ./devops/run.sh recipes.experiment.cogs_v_clips.train \
       trainer.losses.cmpo.enabled=true \
       trainer.losses.cmpo.kl_coef=0.01 \
       trainer.losses.muesli_model.enabled=true \
       trainer.total_timesteps=10000
   ```

4. **Full Muesli** (with dynamics unrolling):
   ```bash
   ./devops/run.sh recipes.experiment.cogs_v_clips.train \
       trainer.losses.dynamics.enabled=true \
       trainer.losses.dynamics.unroll_steps=3 \
       trainer.losses.cmpo.enabled=true \
       trainer.losses.muesli_model.enabled=true \
       trainer.total_timesteps=10000
   ```

Compare training curves and final performance across these configurations.

## References

- Original Muesli paper: Hessel et al., 2021 (https://arxiv.org/abs/2104.06159)
- Implementation: [`muesli.py`](muesli.py), [`cmpo.py`](cmpo.py), [`dynamics.py`](dynamics.py)
- Config: [`losses.py`](losses.py)
