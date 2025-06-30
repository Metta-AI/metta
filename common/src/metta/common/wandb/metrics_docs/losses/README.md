# Losses Metrics

## Overview

Training loss components

**Total metrics in this section:** 8

## Subsections

### General Metrics

**Count:** 8 metrics

**Metric Groups:**
- `approx_kl` (1 metrics)
- `clipfrac` (1 metrics)
- `entropy` (1 metrics)
- `explained_variance` (1 metrics)
- `importance` (1 metrics)
- `l2_reg_loss` (1 metrics)
- `policy_loss` (1 metrics)
- `value_loss` (1 metrics)


## Interpretation Guide

### Loss Components
- `policy_loss` - Actor loss for action selection
- `value_loss` - Critic loss for value estimation
- `entropy` - Policy entropy (exploration)
- `approx_kl` - KL divergence (policy stability)

### Training Health Indicators
1. **Convergence**: Decreasing losses over time
2. **Stability**: Low variance in loss values
3. **Exploration**: Maintain reasonable entropy
4. **Policy Updates**: Monitor `approx_kl` and `clipfrac`
