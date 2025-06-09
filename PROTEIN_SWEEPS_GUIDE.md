# Protein Hyperparameter Sweeps Guide

This guide covers how to run hyperparameter sweeps using the new **Protein optimizer** (pufferlib 3.0's advanced Gaussian Process-based optimizer) that has replaced CARBS in the metta codebase.

## 🚀 Quick Start

### 1. Create a Sweep Configuration

Create a YAML file in `configs/sweep/` with your hyperparameter space:

```yaml
# configs/sweep/my_sweep.yaml
sweep:
  # Optimization settings
  metric: reward  # The metric to optimize
  goal: maximize  # 'maximize' or 'minimize'
  num_random_samples: 10  # Number of random samples before GP optimization

  parameters:
    # Learning rate - log-normal distribution
    trainer.learning_rate:
      min: 1e-5
      max: 1e-1
      mean: 1e-3
      scale: 1
      distribution: log_normal

    # Batch size - integer uniform distribution
    trainer.batch_size:
      min: 16
      max: 128
      mean: 64
      scale: 1
      distribution: int_uniform

    # Gamma - uniform distribution
    trainer.gamma:
      min: 0.8
      max: 1.0
      mean: 0.99
      scale: 0.05
      distribution: uniform
```

### 2. Initialize a Sweep

```bash
# Create and initialize a new sweep
python tools/sweep_init.py \
  sweep_name=my_experiment \
  sweep_params=sweep/my_sweep.yaml \
  wandb.project=my_project \
  wandb.entity=my_team
```

### 3. Run Sweep Evaluations

```bash
# Run the training with sweep parameters
python tools/sweep_eval.py \
  sweep_name=my_experiment \
  wandb.project=my_project \
  wandb.entity=my_team
```

## 📋 Configuration Format

### Parameter Specification

Each parameter in your sweep config must include:

| Field | Description | Required |
|-------|-------------|----------|
| `min` | Minimum value | ✅ |
| `max` | Maximum value | ✅ |
| `mean` | Center/mean value for sampling | ✅ |
| `scale` | Scale parameter for distribution | ✅ |
| `distribution` | Distribution type | ✅ |

### Supported Distributions

| Distribution | Description | Example Use Case |
|-------------|-------------|-----------------|
| `uniform` | Uniform distribution | General numeric ranges, probabilities |
| `int_uniform` | Integer uniform distribution | Batch sizes, epochs, discrete choices |
| `uniform_pow2` | Power-of-2 uniform distribution | Architecture dimensions (powers of 2) |
| `log_normal` | Log-normal distribution | Learning rates, regularization coefficients |
| `logit_normal` | Logit-normal distribution | Bounded probabilities (0,1) |

**Note**: `normal` distribution is not supported - use `uniform` for similar behavior.

### Example Configurations

#### Simple Configuration
```yaml
# configs/sweep/simple_sweep.yaml
sweep:
  learning_rate:
    min: 1e-5
    max: 1e-2
    scale: 1
    mean: 1e-3
    distribution: log_normal

  batch_size:
    min: 16
    max: 128
    scale: 1
    mean: 64
    distribution: int_uniform
```

#### Complex Configuration
```yaml
# configs/sweep/advanced_sweep.yaml
sweep:
  metric: action.use.altar
  goal: maximize
  num_random_samples: 10

  parameters:
    # Environment parameters
    env.sampling:
      min: 0.0
      max: 1.0
      mean: 0.001
      scale: 1
      distribution: uniform

    # PPO parameters
    trainer.learning_rate:
      min: 1e-5
      max: 1e-1
      mean: 1e-3
      scale: 1
      distribution: log_normal

    trainer.gamma:
      min: 0.0
      max: 1.0
      mean: 0.99
      scale: 0.1
      distribution: uniform

    trainer.gae_lambda:
      min: 0.0
      max: 1.0
      mean: 0.95
      scale: 0.1
      distribution: uniform

    trainer.clip_coef:
      min: 0.0
      max: 1.0
      mean: 0.2
      scale: 0.1
      distribution: uniform

    # Architecture parameters
    trainer.batch_size:
      min: 65536
      max: 524288
      mean: 131072
      scale: 1
      distribution: int_uniform

    trainer.bptt_horizon:
      min: 1
      max: 128
      mean: 32
      scale: 1
      distribution: int_uniform
```

## 🔧 Complete Workflow

### Step 1: Create Sweep Configuration
```bash
# Create your sweep config
cat > configs/sweep/my_experiment.yaml << EOF
sweep:
  metric: reward
  goal: maximize
  num_random_samples: 5

  parameters:
    trainer.learning_rate:
      min: 1e-5
      max: 1e-1
      mean: 1e-3
      scale: 1
      distribution: log_normal

    trainer.batch_size:
      min: 32
      max: 256
      mean: 128
      scale: 1
      distribution: int_uniform
EOF
```

### Step 2: Initialize Sweep
```bash
# This creates the sweep in WandB and generates the first suggestion
python tools/sweep_init.py \
  sweep_name=my_experiment_v1 \
  sweep_params=sweep/my_experiment.yaml \
  wandb.project=metta_sweeps \
  wandb.entity=your_team \
  +runs_dir=./runs/my_experiment_v1
```

This will:
- Create a WandB sweep
- Initialize the Protein optimizer with your parameter space
- Generate the first hyperparameter suggestion
- Save configuration files to the run directory

### Step 3: Run Training with Sweep Parameters
```bash
# This runs training with the suggested parameters and reports results
python tools/sweep_eval.py \
  sweep_name=my_experiment_v1 \
  wandb.project=metta_sweeps \
  wandb.entity=your_team \
  +runs_dir=./runs/my_experiment_v1
```

This will:
- Load the hyperparameter suggestion
- Run training with those parameters
- Report results (objective and cost) back to the Protein optimizer
- Update the Gaussian Process model for better future suggestions

### Step 4: Continue the Sweep
```bash
# Run more iterations - each will get better suggestions based on previous results
for i in {2..10}; do
  python tools/sweep_init.py sweep_name=my_experiment_v1 # Get next suggestion
  python tools/sweep_eval.py sweep_name=my_experiment_v1 # Evaluate it
done
```

## 🎯 Advanced Features

### Protein Optimizer Benefits

The **Protein optimizer** provides several advantages over the previous CARBS system:

1. **Gaussian Process Optimization**: Uses sophisticated GP models to predict performance
2. **Pareto Frontier Analysis**: Considers both objective value and computational cost
3. **Multi-objective Optimization**: Balances performance vs. efficiency
4. **Adaptive Sampling**: Learns from previous experiments to suggest better parameters
5. **Cost-aware Suggestions**: Considers training time/cost in recommendations

### WandB Integration Features

- **State Tracking**: Tracks optimizer state (`initializing`, `running`, `success`, `failure`)
- **Previous Run Loading**: Automatically loads and learns from previous sweep runs
- **Suggestion UUIDs**: Tracks which suggestions belong to which runs
- **Failure Handling**: Properly handles and learns from failed runs
- **Cost Tracking**: Records both objective values and computational costs

### Monitoring Your Sweep

In WandB, you can monitor:

- **Hyperparameter Suggestions**: See what the optimizer is trying
- **Performance Trends**: Track how suggestions improve over time
- **Cost vs. Performance**: Analyze the Pareto frontier
- **Convergence**: Watch the optimizer learn and focus on promising regions

## 🐛 Troubleshooting

### Common Issues

#### 1. "Run already has protein state" Error
```bash
# Clear the sweep state if needed
rm -rf runs/your_sweep_name
```

#### 2. Invalid Distribution Error
Make sure your distribution names are correct:
- ✅ `log_normal` (not `log-normal`)
- ✅ `int_uniform` (not `int_uniform_distribution`)
- ✅ `uniform` (not `normal` - use `uniform` instead)
- ✅ `logit_normal` (for bounded probabilities)
- ✅ `uniform_pow2` (for power-of-2 values)

#### 3. Parameter Space Issues
Ensure your parameter ranges make sense:
```yaml
# ❌ Bad: min > max
learning_rate:
  min: 1e-1
  max: 1e-5  # Error!

# ✅ Good: min < max
learning_rate:
  min: 1e-5
  max: 1e-1
```

#### 4. WandB Authentication
```bash
# Login to WandB if needed
wandb login
```

### Debugging Commands

```bash
# Check sweep configuration
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/sweep/my_sweep.yaml')
print(OmegaConf.to_yaml(cfg))
"

# Test Protein initialization
python -c "
from metta.rl.carbs.metta_protein import MettaProtein
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/sweep/my_sweep.yaml')
protein = MettaProtein(cfg)
print('✅ Protein initialized successfully')
suggestion, info = protein.suggest()
print(f'First suggestion: {suggestion}')
"
```

## 📊 Example Results

After running a sweep, you should see the Protein optimizer learning and improving:

```
Iteration 1: Score: 0.456 Cost: 150.2 (Random exploration)
Iteration 2: Score: 0.523 Cost: 142.8 (Random exploration)
Iteration 3: Score: 0.601 Cost: 135.4 (GP-guided suggestion)
Iteration 4: Score: 0.678 Cost: 128.9 (GP-guided suggestion)
Iteration 5: Score: 0.742 Cost: 118.3 (GP-guided suggestion)
...
```

The "Predicted" lines show the Gaussian Process predictions:
```
Predicted --  Score: 0.750 Cost: 120.500 Rating: 0.892
```

This indicates the optimizer is learning the parameter space and making informed suggestions!

## 🔗 Related Files

- **Sweep Configurations**: `configs/sweep/*.yaml`
- **Protein Implementation**: `metta/rl/carbs/metta_protein.py`
- **Sweep Tools**: `tools/sweep_init.py`, `tools/sweep_eval.py`
- **Tests**: `tests/rl/test_wandb_protein.py`, `tests/rl/test_protein_integration.py`

---

**🎉 You're now ready to run sophisticated hyperparameter sweeps with the Protein optimizer!**

The system will automatically learn from your experiments and suggest increasingly better hyperparameters, leading to faster convergence and better final performance.
