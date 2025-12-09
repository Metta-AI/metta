# Muesli Algorithm Implementation

This directory contains the implementation of the Muesli (Model-Based Policy Optimization) algorithm from [Hessel et al., 2021](https://arxiv.org/abs/2104.06159).

## Overview

Muesli combines model-based and model-free RL by training a dynamics model alongside the policy. It consists of two main components:

1. **Model Loss (`muesli.py`)**: Trains the dynamics network to predict future values, rewards, and policies through K-step unrolling
2. **CMPO Loss (`cmpo.py`)**: Conservative Model-based Policy Optimization that provides stable target policies

## Components

### 1. Muesli Model Loss (L_m)

The model loss trains a dynamics network to predict future states that yield accurate value, reward, and policy predictions:

```
L_m = Σ_{k=1}^{K} [ L_v^k + L_r^k + L_π^k ]
```

Where:
- **L_v^k**: Value prediction loss (MSE between predicted and target values)
- **L_r^k**: Reward prediction loss (MSE between predicted and actual rewards)
- **L_π^k**: Policy prediction loss (cross-entropy with target policy)

### 2. CMPO (Conservative Model-based Policy Optimization)

CMPO provides a stable target policy by reweighting the prior policy with clipped advantages:

```
π_cmpo(a|s) ∝ π_prior(a|s) * exp(clip(Â(s,a) / σ²))
```

This creates a conservative policy improvement that's more stable than raw advantage-weighted updates.

## Usage

### Basic Configuration

```python
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig

# Enable Muesli components
trainer_cfg = TrainerConfig(
    losses=LossesConfig(
        # Keep standard PPO components
        ppo_actor=PPOActorConfig(enabled=True),
        ppo_critic=PPOCriticConfig(enabled=True),

        # Add Muesli components
        cmpo=CMPOConfig(
            enabled=True,
            kl_coef=0.1,  # KL regularization strength
            advantage_clip_min=-1.0,
            advantage_clip_max=1.0,
        ),
        muesli_model=MuesliModelConfig(
            enabled=True,
            unroll_steps=5,  # K-step unrolling
            value_pred_coef=1.0,
            reward_pred_coef=1.0,
            policy_pred_coef=1.0,
            dynamics_grad_scale=0.5,  # Gradient scaling for dynamics
        ),
    )
)
```

### Using with CoGs vs Clips

```python
from recipes.experiment.cogs_v_clips import train

# Create training tool
tt = train(num_cogs=4, mission="easy_hearts")

# Enable Muesli
tt.trainer.losses.cmpo.enabled = True
tt.trainer.losses.cmpo.kl_coef = 0.1

tt.trainer.losses.muesli_model.enabled = True
tt.trainer.losses.muesli_model.unroll_steps = 5

# Run training
await tt.run(run="muesli_experiment")
```

### Training with Muesli

```bash
# Train with Muesli enabled
uv run ./tools/run.py train arena run=muesli_test \
    trainer.total_timesteps=100000 \
    trainer.losses.cmpo.enabled=true \
    trainer.losses.cmpo.kl_coef=0.1 \
    trainer.losses.muesli_model.enabled=true \
    trainer.losses.muesli_model.unroll_steps=5
```

## Policy Requirements

For Muesli to work, your policy must output the following in its forward pass:

### Required Outputs

```python
class MuesliCompatiblePolicy(Policy):
    def forward(self, td: TensorDict, **kwargs) -> TensorDict:
        # ... forward pass logic ...

        td["hidden_state"] = h  # Latent representation (B, T, H)
        td["value_pred"] = v  # Value prediction (B, T, 1)
        td["logits"] = logits  # Policy logits (B, T, A)
        td["reward_pred"] = r  # Reward prediction (B, T, 1)

        return td
```

### Alternative: Dedicated Components

You can also implement dedicated components:
- `representation_net`: obs → h_0 (initial hidden state)
- `dynamics_net`: (h, a) → (h', r_pred) (dynamics model)
- `prediction_net`: h → (value, logits) (prediction heads)

## Configuration Parameters

### MuesliModelConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `unroll_steps` | 5 | Number of steps to unroll the dynamics model |
| `value_pred_coef` | 1.0 | Weight for value prediction loss |
| `reward_pred_coef` | 1.0 | Weight for reward prediction loss |
| `policy_pred_coef` | 1.0 | Weight for policy prediction loss |
| `dynamics_grad_scale` | 0.5 | Gradient scaling for dynamics network |
| `use_target_network` | True | Whether to use target network for stable targets |
| `target_ema_decay` | 0.99 | EMA decay for target network (τ) |

### CMPOConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kl_coef` | 0.1 | KL regularization coefficient |
| `advantage_clip_min` | -1.0 | Minimum advantage clip value |
| `advantage_clip_max` | 1.0 | Maximum advantage clip value |
| `use_target_network` | True | Whether to use target network for prior policy |
| `target_ema_decay` | 0.99 | EMA decay for target network |
| `min_advantage_variance` | 1e-8 | Minimum variance for advantage normalization |

## Loss Ordering

The losses execute in this order (defined in [`losses.py`](losses.py)):

1. `ppo_critic` - Value function training
2. **`cmpo`** - Compute CMPO target policies ← _Muesli component_
3. `ppo_actor` - Policy gradient update
4. **`muesli_model`** - Model loss with K-step unrolling ← _Muesli component_

CMPO runs before the model loss so that CMPO target policies are available for the model's policy prediction loss.

## Tracked Metrics

### Muesli Model Loss

- `muesli_model_loss`: Total model loss
- `muesli_value_loss`: Value prediction component
- `muesli_reward_loss`: Reward prediction component
- `muesli_policy_loss`: Policy prediction component

### CMPO Loss

- `cmpo_loss`: Total CMPO loss
- `cmpo_kl_loss`: KL divergence component
- `cmpo_adv_mean`: Mean advantage value
- `cmpo_adv_std`: Advantage standard deviation
- `cmpo_adv_variance_ema`: EMA of advantage variance

## Implementation Notes

### Gradient Scaling

The paper recommends scaling gradients to the dynamics network by 0.5 to stabilize training. This is controlled by `dynamics_grad_scale`.

### Target Networks

Both components use exponential moving average (EMA) target networks for stable value/policy targets. The decay rate (τ) controls how quickly the target tracks the online network.

### CMPO as Policy Target

CMPO provides principled target policies for the model's policy prediction loss. This is more stable than using the current policy directly.

### K-Step Unrolling

The model loss unrolls the dynamics for K steps, predicting:
- **Step 0**: Use representations from policy forward
- **Steps 1..K**: Unroll dynamics: h_{t+k}, r_k = f_dynamics(h_{t+k-1}, a_{t+k-1})

## References

1. **Muesli Paper**: [Hessel et al., "Muesli: Combining Improvements in Policy Optimization", 2021](https://arxiv.org/abs/2104.06159)
2. **MuZero**: Original dynamics model framework
3. **IMPALA**: V-trace and off-policy corrections
4. **PPO**: Base policy optimization algorithm

## Testing

Run the test suite:

```bash
uv run pytest tests/rl/test_muesli_losses.py -v
```

The tests cover:
- ✓ Initialization of both loss components
- ✓ Target network updates (EMA)
- ✓ CMPO policy computation
- ✓ Loss forward passes
- ✓ Integration between components
