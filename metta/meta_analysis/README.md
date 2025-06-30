# Meta-Analysis for Training Curve Prediction

This module implements a meta-analysis system that can predict training curves (reward vs timesteps) from environment and agent configurations using Variational Autoencoders (VAEs).

## Overview

The system consists of three main components:

1. **Environment VAE**: Compresses environment configurations into a latent space
2. **Agent VAE**: Compresses agent hyperparameters into a latent space
3. **Reward Predictor**: Predicts training curves from the combined latent representations

## Architecture

```
Environment Config → Environment VAE → Environment Latent
                                    ↓
Reward Predictor ← Combined Latents ← Agent Latent ← Agent VAE ← Agent Config
                                    ↑
Training Curve ← Predicted Curve
```

## Key Features

- **Latent Space Learning**: VAEs learn compressed representations of environment and agent configurations
- **Training Curve Prediction**: Predicts full reward curves (hearts vs timesteps) from configs
- **Wandb Integration**: Automatically collects training data from existing wandb runs
- **Configurable**: Supports various environment and agent parameters

## Usage

### Quick Demo

Run the demonstration script to see the system in action:

```bash
./tools/meta_analysis_demo.py collect_data=true train_model=true demo_predictions=true
```

### Data Collection

The system can automatically collect training data from your existing wandb runs:

```python
from metta.meta_analysis import TrainingDataCollector

collector = TrainingDataCollector("your-entity", "your-project")
training_data = collector.collect_training_runs(max_runs=100)
```

### Training

Train the meta-analysis model:

```python
from metta.meta_analysis import MetaAnalysisModel, MetaAnalysisTrainer, TrainingCurveDataset

# Create model
model = MetaAnalysisModel(
    env_input_dim=9,  # Number of environment features
    agent_input_dim=12,  # Number of agent features
    env_latent_dim=16,
    agent_latent_dim=16,
    curve_length=100
)

# Create trainer
trainer = MetaAnalysisTrainer(model, device="cpu")

# Train
trainer.train(train_dataloader, val_dataloader, num_epochs=100)
```

### Predictions

Generate predictions for new configurations:

```python
# Predict curve for specific configs
env_config = torch.tensor([...])  # Environment parameters
agent_config = torch.tensor([...])  # Agent parameters
predicted_curve = trainer.predict_curve(env_config, agent_config)

# Sample from latent space
env_latent, agent_latent = trainer.sample_latent_space(num_samples=10)
predicted_curves = trainer.model.predict_curve(env_latent, agent_latent)
```

## Environment Parameters

The system extracts these environment parameters:

- `max_steps`: Maximum steps per episode
- `num_agents`: Number of agents in environment
- `map_width/height`: Environment size
- `num_rooms`: Number of rooms
- `num_altars/mines/generators/walls`: Object counts

## Agent Parameters

The system extracts these agent parameters:

- `learning_rate`: Learning rate
- `batch_size/minibatch_size`: Batch sizes
- `gamma/gae_lambda`: RL parameters
- `clip_coef/ent_coef/vf_coef`: Loss coefficients
- `hidden_size/num_layers`: Network architecture
- `cnn_channels/kernel_size`: CNN parameters

## Configuration

Adjust the demo configuration in `configs/meta_analysis_demo.yaml`:

```yaml
# Model architecture
env_latent_dim: 16
agent_latent_dim: 16
curve_length: 100
hidden_dim: 128

# Training settings
batch_size: 8
learning_rate: 0.001
beta: 1.0  # VAE KL loss weight
curve_weight: 1.0  # Reward prediction loss weight
num_epochs: 50
```

## Future Work

- **Curriculum Generation**: Use latent space to generate curriculum sequences
- **Difficulty Estimation**: Analyze what makes tasks difficult
- **Hyperparameter Optimization**: Use predictions to guide hyperparameter search
- **Multi-task Learning**: Extend to multiple task types
- **Interpretability**: Analyze what the latent dimensions represent

## Requirements

- PyTorch
- Wandb
- Pandas
- NumPy
- Hydra (for configuration)

## Integration with Existing Codebase

This system integrates with your existing:
- Wandb artifacts and runs
- Environment configurations
- Agent hyperparameters
- Training curves and metrics

The data collector automatically extracts relevant information from your existing training runs, making it easy to leverage your current data for meta-analysis.
