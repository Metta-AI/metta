# Losses Metrics

## Overview

Training loss components that indicate learning progress and stability. Monitor these to
ensure healthy training dynamics.

**Total metrics in this section:** 9

## Subsections

### General Metrics

**Count:** 9 metrics

**approx_kl:** (1 value)
- `losses/approx_kl`
  - Approximate KL divergence between old and new policies.
  - **Interpretation:** Should stay below target threshold (typically 0.01-0.02). High values trigger early stopping.


**clipfrac:** (1 value)
- `losses/clipfrac`
  - Fraction of samples clipped by PPO's objective function.
  - **Interpretation:** Typically 0.1-0.3. Very high values suggest too large policy updates.


**current_logprobs:** (1 value)
- `losses/current_logprobs`

**entropy:** (1 value)
- `losses/entropy`
  - Policy entropy measuring action distribution randomness.
  - **Interpretation:** Higher entropy encourages exploration. Should gradually decrease but not reach zero.


**explained_variance:** (1 value)
- `losses/explained_variance`

**importance:** (1 value)
- `losses/importance`

**l2_reg_loss:** (1 value)
- `losses/l2_reg_loss`

**policy_loss:** (1 value)
- `losses/policy_loss`
  - Actor network loss measuring action prediction quality.
  - **Interpretation:** Should decrease over time but not to zero. Sudden spikes may indicate instability.


**value_loss:** (1 value)
- `losses/value_loss`
  - Critic network loss measuring value prediction accuracy.
  - **Interpretation:** Lower is better, but some noise is expected. High values suggest poor value estimates.



