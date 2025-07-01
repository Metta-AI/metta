# Losses Metrics

## Overview

Training loss components that indicate learning progress and stability. Monitor these to
ensure healthy training dynamics.

**Total metrics in this section:** 8

## Subsections

### General Metrics

**Count:** 8 metrics

**approx_kl:**
- `losses/approx_kl`
  Approximate KL divergence between old and new policies.
  **Interpretation:** Should stay below target threshold (typically 0.01-0.02). High values trigger early stopping.


**clipfrac:**
- `losses/clipfrac`
  Fraction of samples clipped by PPO's objective function.
  **Interpretation:** Typically 0.1-0.3. Very high values suggest too large policy updates.


**entropy:**
- `losses/entropy`
  Policy entropy measuring action distribution randomness.
  **Interpretation:** Higher entropy encourages exploration. Should gradually decrease but not reach zero.


**explained_variance:**
- `losses/explained_variance`

**importance:**
- `losses/importance`

**l2_reg_loss:**
- `losses/l2_reg_loss`

**policy_loss:**
- `losses/policy_loss`
  Actor network loss measuring action prediction quality.
  **Interpretation:** Should decrease over time but not to zero. Sudden spikes may indicate instability.


**value_loss:**
- `losses/value_loss`
  Critic network loss measuring value prediction accuracy.
  **Interpretation:** Lower is better, but some noise is expected. High values suggest poor value estimates.



