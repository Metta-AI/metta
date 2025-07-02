# VAPOR: Variational Policy Optimization

Implementation of VAPOR (Variational Policy Optimization) from "Probabilistic Inference in Reinforcement Learning Done Right" (Tarbouriech et al., 2023).

## Overview

VAPOR reformulates reinforcement learning as **variational inference over state-action optimality**. Instead of standard policy gradient methods like PPO, VAPOR maintains a variational approximation to the posterior probability that each state-action pair is optimal.

### Key Benefits

1. **Principled Exploration**: Uses posterior uncertainty for exploration instead of random noise
2. **Stable Updates**: Variational inference framework provides more stable policy updates
3. **Better Sample Efficiency**: Often requires fewer samples than standard PPO
4. **Theoretical Grounding**: Solid mathematical foundation in Bayesian inference

### How It Works

1. **Posterior Approximation**: Maintains a variational approximation to P(optimal | state, action)
2. **Weighted Updates**: Policy gradients are weighted by posterior optimality probabilities
3. **Uncertainty Exploration**: Exploration bonus based on posterior entropy
4. **KL Regularization**: Prevents policy from changing too quickly (controlled by β parameter)

## Usage

### Basic Usage

```bash
# Train with VAPOR instead of PPO
./tools/train.py trainer=vapor run=my_vapor_experiment

# Train with specific hardware configuration
./tools/train.py trainer=vapor run=my_vapor_experiment +hardware=macbook

# Override VAPOR parameters
./tools/train.py trainer=vapor run=high_exploration \
  trainer.ppo.vapor.beta=2.0 \
  trainer.ppo.vapor.exploration_bonus=0.2

# Use the example script
python examples/train_with_vapor.py

# Quick config validation (dry-run)
./tools/train.py trainer=vapor run=vapor_test --cfg job
```

### Validation

To verify VAPOR is configured correctly, check that the config shows:
```yaml
ppo:
  use_vapor: true
  vapor:
    beta: 1.0
    beta_schedule: linear
    # ... other vapor settings
```

### Configuration

VAPOR is configured through the `trainer.ppo.vapor` section:

```yaml
ppo:
  use_vapor: true
  vapor:
    beta: 1.0                    # KL regularization strength (temperature)
    beta_schedule: "linear"      # "constant", "linear", or "exponential"
    min_beta: 0.1               # Minimum beta during annealing
    exploration_bonus: 0.1       # Uncertainty-based exploration coefficient
    use_importance_weighting: true  # Use importance sampling for off-policy correction
```

### Key Parameters

#### Beta (β) - Temperature Parameter
- **High β (1.0+)**: More exploration, less focused updates
- **Low β (0.1-0.5)**: More exploitation, more focused updates
- **Scheduling**: Usually start high and anneal down during training

#### Beta Schedule
- `"constant"`: Fixed β throughout training
- `"linear"`: Linear annealing from β to min_β
- `"exponential"`: Exponential decay from β to min_β

#### Exploration Bonus
- Controls how much to bonus uncertain state-actions
- Higher values → more exploration
- Typical range: 0.05 - 0.2

## Monitoring VAPOR Training

VAPOR adds these metrics to training logs:

```
vapor_policy_loss          # VAPOR policy gradient loss
vapor_kl_penalty           # KL regularization penalty
vapor_exploration_bonus    # Uncertainty-based exploration bonus
vapor_posterior_entropy    # Entropy of optimality posterior (exploration measure)
vapor_beta                 # Current β value (anneals during training)
```

### Expected Behavior

- **vapor_posterior_entropy**: Should be high early (exploring) and decrease as policy improves
- **vapor_beta**: Should anneal from initial value to min_beta if using scheduling
- **vapor_exploration_bonus**: Should decrease as uncertainty decreases
- **vapor_kl_penalty**: Prevents too-rapid policy changes

## Comparison with PPO

| Aspect | PPO | VAPOR |
|--------|-----|-------|
| **Updates** | Clipped policy gradients | Posterior-weighted gradients |
| **Exploration** | Random noise (entropy) | Uncertainty-based exploration |
| **Stability** | Clipping mechanism | KL regularization |
| **Theory** | Heuristic clipping | Principled variational inference |
| **Sample Efficiency** | Good | Often better |

## Implementation Details

### Core Algorithm

1. **Rollout Collection**: Same as PPO - collect experiences
2. **Advantage Estimation**: Same GAE computation
3. **Posterior Approximation**: Convert advantages to optimality probabilities
4. **Weighted Policy Loss**: Weight policy gradients by posterior probabilities
5. **KL Regularization**: Add β * KL(π_new || π_old) penalty
6. **Exploration Bonus**: Add uncertainty-based exploration term

### Code Structure

- `metta/rl/vapor_losses.py`: Core VAPOR loss implementation
- `metta/rl/functions.py`: Integration with training loop
- `metta/rl/trainer_config.py`: Configuration schema
- `configs/trainer/vapor.yaml`: Default VAPOR configuration

## Experimental Tips

### Hyperparameter Tuning

1. **Start with β=1.0**: Good default for most environments
2. **Use Linear Annealing**: `beta_schedule="linear"` works well
3. **Moderate Exploration**: `exploration_bonus=0.1` is usually good
4. **Monitor Entropy**: Watch `vapor_posterior_entropy` for exploration

### Troubleshooting

**Poor Exploration**:
- Increase β or exploration_bonus
- Use linear/exponential β annealing

**Unstable Training**:
- Decrease β or increase min_beta
- Reduce exploration_bonus

**Slow Learning**:
- Increase β early in training
- Check that β annealing isn't too aggressive

### Multi-Agent Considerations

VAPOR works particularly well in multi-agent settings because:
- Posterior uncertainty captures complex interaction dynamics
- Principled exploration helps discover coordination strategies
- Less sensitive to opponent policy changes than PPO

## Theory Background

VAPOR is based on the insight that RL can be formulated as inference over a graphical model where we want to infer which state-action pairs are optimal. The variational approximation provides:

1. **Tractable Inference**: Converts intractable posterior to tractable optimization
2. **Exploration**: Uncertainty in posterior drives exploration
3. **Stability**: KL regularization provides natural policy constraint
4. **Convergence**: Principled objective with convergence guarantees

For full theoretical details, see the original paper: "Probabilistic Inference in Reinforcement Learning Done Right" (Tarbouriech et al., 2023).

## Examples

### Simple VAPOR Training
```bash
# Basic VAPOR training
./tools/train.py trainer=vapor run=vapor_test

# With multiple workers
./tools/train.py trainer=vapor run=vapor_test trainer.num_workers=4
```

### Custom VAPOR Configuration
```bash
# High exploration VAPOR
./tools/train.py trainer=vapor run=high_exploration \
  trainer.ppo.vapor.beta=2.0 \
  trainer.ppo.vapor.exploration_bonus=0.2 \
  trainer.ppo.vapor.beta_schedule=exponential

# Conservative VAPOR with linear annealing
./tools/train.py trainer=vapor run=conservative_vapor \
  trainer.ppo.vapor.beta=0.5 \
  trainer.ppo.vapor.beta_schedule=linear \
  trainer.ppo.vapor.min_beta=0.1

# Quick test with small batch size
./tools/train.py trainer=vapor run=vapor_quick_test \
  +hardware=macbook \
  trainer.num_workers=1 \
  trainer.total_timesteps=10000 \
  trainer.batch_size=512 \
  trainer.minibatch_size=512 \
  wandb=off
```

### Compare with PPO
```bash
# Train PPO baseline
./tools/train.py trainer=trainer run=ppo_baseline

# Train VAPOR with same setup
./tools/train.py trainer=vapor run=vapor_compare

# Direct comparison with same parameters
./tools/train.py trainer=trainer run=ppo_baseline trainer.total_timesteps=1000000
./tools/train.py trainer=vapor run=vapor_baseline trainer.total_timesteps=1000000
```

### Advanced Usage
```bash
# VAPOR with prioritized experience replay
./tools/train.py trainer=vapor run=vapor_per \
  trainer.prioritized_experience_replay.prio_alpha=0.6 \
  trainer.ppo.vapor.beta=1.5

# VAPOR with custom learning rate and batch sizes
./tools/train.py trainer=vapor run=vapor_tuned \
  trainer.optimizer.learning_rate=0.001 \
  trainer.batch_size=2048 \
  trainer.minibatch_size=512 \
  trainer.ppo.vapor.beta_schedule=exponential

# Debug run with verbose output
HYDRA_FULL_ERROR=1 ./tools/train.py trainer=vapor run=vapor_debug \
  trainer.verbose=true \
  trainer.num_workers=1 \
  trainer.total_timesteps=1000
```

The VAPOR implementation in Metta provides a powerful alternative to standard PPO that often shows improved sample efficiency and more principled exploration, especially in complex multi-agent environments.
