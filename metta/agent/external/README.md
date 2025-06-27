# External PyTorch Policies

This directory contains utilities for loading external PyTorch policies (e.g., from PufferLib) into Metta. This allows you to train and evaluate policies that were developed outside of Metta's standard agent architecture.

## Directory Contents

- `example.py` - Reference implementation showing how to adapt a PufferLib-style policy for Metta
- `lstm_transformer.py` - Alternative LSTM-Transformer hybrid architecture example

## Overview

External policies are loaded using the `pytorch://` URI scheme. These policies are automatically wrapped in a `PytorchAgent` adapter that translates between the external policy interface and Metta's expected interface.

## Quick Start

### 1. Training with an External Policy

```bash
python tools/train.py \
    hydra.run.dir=runs/external_policy \
    policy_uri=pytorch://path/to/your/policy.pt \
    hardware=local_debug
```

### 2. Evaluating an External Policy

```bash
python tools/sim.py \
    hydra.run.dir=runs/eval_external \
    policy_uri=pytorch://path/to/your/policy.pt \
    env.mettagrid.curriculum.test=mettagrid/memory/memory_one_apple
```

## URI Schemes

Metta supports multiple URI schemes for loading policies:

- `pytorch://path/to/policy.pt` - Load external PyTorch checkpoint
- `file://path/to/metta/policy.pt` - Load Metta-native policy
- `wandb://entity/project/artifact/name` - Load from Weights & Biases
- `/path/to/policy.pt` - Defaults to `file://` scheme

## Creating an External Policy

External policies must follow a specific interface to be compatible with Metta. See `example.py` for a complete implementation. Key requirements:

### 1. Policy Structure

```python
class Policy(nn.Module):
    def __init__(self, env, **kwargs):
        super().__init__()
        # Initialize your network architecture

    def forward(self, observations, state=None):
        # Process observations and return (actions, value), hidden_state
        pass
```

### 2. Recurrent Wrapper (if using LSTM)

```python
class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, **kwargs):
        if policy is None:
            policy = Policy(env, hidden_size=hidden_size, **kwargs)
        super().__init__(env, policy, input_size, hidden_size)
```

### 3. Required Methods

- `encode_observations(observations, state)` - Process raw observations into features
- `decode_actions(hidden)` - Convert hidden state to action logits and value
- `forward(observations, state)` - Main forward pass

## Configuration

External policies can be configured via Hydra configs:

```yaml
# In your config file
policy_uri: pytorch://checkpoints/external_model.pt

# Optional: Configure the external policy loading
pytorch:
  _target_: metta.agent.external.example.Recurrent
  hidden_size: 512
  cnn_channels: 128
```

## Integration with Metta Features

External policies automatically benefit from Metta's features:

- **Feature Remapping**: Handles environments where feature IDs differ from training
- **Action Filtering**: Supports dynamic action spaces
- **State Management**: Maintains LSTM/hidden states across episodes
- **Distributed Training**: Works with Metta's distributed training infrastructure

## Example: Loading a PufferLib Policy

If you have a policy trained with PufferLib:

```python
# Your PufferLib training produced a checkpoint
checkpoint_path = "puffer_metta.pt"

# Use it in Metta for evaluation
python tools/sim.py \
    policy_uri=pytorch://${checkpoint_path} \
    sim.episodes=100 \
    sim.render=true
```

## Debugging Tips

1. **Check Policy Loading**: Enable debug logging to see policy loading details:
   ```bash
   export HYDRA_FULL_ERROR=1
   export LOGLEVEL=DEBUG
   ```

2. **Verify Architecture**: The policy architecture must match the checkpoint. Common issues:
   - Mismatched hidden sizes
   - Different number of layers
   - Incompatible observation/action spaces

3. **State Compatibility**: Ensure your policy handles Metta's `PolicyState` object correctly

## Understanding Observation Format

Metta uses tokenized observations in the format `[location, feature_id, value]` where each observation is a 3-byte token:
- **location** (byte 0): Encodes x,y coordinates (4 bits each)
- **feature_id** (byte 1): The feature type (e.g., wall=0, agent=1, mineral=3)
- **value** (byte 2): The feature value (e.g., health amount, resource count)

The `example.py` policy shows how to decode these tokens into the standard grid format expected by most neural networks.

## Advanced Usage

### Custom External Policies

To create your own external policy adapter:

1. Create a new file in `metta/agent/external/`
2. Implement the required interface (see `example.py`)
3. Update your config to use your custom policy:
   ```yaml
   pytorch:
     _target_: metta.agent.external.your_policy.YourRecurrent
   ```

### Hybrid Training

You can start with an external policy and continue training in Metta:

```bash
# Continue training from external checkpoint
python tools/train.py \
    policy_uri=pytorch://external_checkpoint.pt \
    trainer.total_timesteps=10_000_000
```

The trained policy will be saved in Metta's native format for future use.

### Converting Between Formats

If you need to convert a Metta policy to use elsewhere:

```python
# Load a Metta policy
from metta.agent.policy_store import PolicyStore
store = PolicyStore(cfg, wandb_run=None)
pr = store.load_from_uri("file://path/to/metta_policy.pt")

# Access the underlying PyTorch model
model = pr.policy.policy  # The inner policy module
torch.save(model.state_dict(), "exported_weights.pt")
```
