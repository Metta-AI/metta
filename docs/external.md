# External PyTorch Policies

This documentation covers utilities for loading external PyTorch policies (e.g., from PufferLib) into Metta. This allows you to train and evaluate policies that were developed outside of Metta's standard agent architecture.

## External Loading Code Location

The external policy loading implementation is located in `metta/agent/external/`:

- `metta/agent/external/pytorch_adapter.py` - Unified `PytorchAdapter` class that wraps external policies for MettaAgent compatibility
- `metta/agent/external/torch.py` - Exact copy of PufferLib's torch.py policy (unmodified) - works perfectly with our adapter
- `metta/agent/external/lstm_transformer.py` - Alternative LSTM-Transformer hybrid architecture with Metta-specific enhancements

## Overview

External policies are loaded using the `pytorch://` URI scheme. These policies are automatically wrapped in a `PytorchAdapter` that translates between the external policy interface and Metta's expected interface. The adapter handles:

- Different naming conventions (critic→value, hidden→logits)
- State management for LSTM policies and state conversion
- Token observation handling [B, M, 3]
- PufferLib LSTMWrapper patterns
- Method forwarding for MettaAgent compatibility

The key advantage is that external policies can be used **without modification**. The `torch.py` file is an exact copy from PufferLib, and all necessary conversions are handled by the adapter.

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

External policies must follow a specific interface to be compatible with Metta. See `metta/agent/external/torch.py` for a working example. Key requirements:

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
  _target_: metta.agent.external.torch.Recurrent  # Or your custom policy class
  hidden_size: 512
  cnn_channels: 128
```

### Using the Adapter System

The `PytorchAdapter` class (in `metta/agent/external/pytorch_adapter.py`) automatically detects the type of external policy and applies the appropriate conversions:

- For PufferLib LSTMWrapper policies (like `torch.Recurrent`), it handles the LSTM state management and forward_train conversions
- For standard PyTorch policies, it provides basic interface translation
- Backwards compatibility: `ExternalPolicyAdapter` is kept as an alias

Example configuration for different policy types:

```yaml
# For PufferLib policies (torch.py) - can be used without modification
pytorch:
  _target_: metta.agent.external.torch.Recurrent
  hidden_size: 512

# For LSTM-Transformer hybrid
pytorch:
  _target_: metta.agent.external.lstm_transformer.Recurrent
  hidden_size: 384
  depth: 3
  num_heads: 6

# For custom external policies
pytorch:
  _target_: path.to.your.ExternalPolicy
  custom_arg: value
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

The `torch.py` policy (see the `encode_observations` method in `metta/agent/external/torch.py`) shows how to decode these tokens into the standard grid format expected by CNNs.

## Advanced Usage

### Custom External Policies

To create your own external policy adapter:

1. Create a new file in `metta/agent/external/`
2. Implement the required interface (see `metta/agent/external/torch.py` for an example)
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

## Token Observations in Metta

Metta uses token observations natively with shape `[B, M, 3]` where:
- M is the number of observation tokens (configurable, typically 200)
- Each token has 3 channels: [location_byte, feature_id, feature_value]
- The location byte encodes x,y coordinates within the observation window (high 4 bits = row, low 4 bits = column)

The unmodified PufferLib policies (like `torch.py`) are designed to work with exactly this token format. They internally convert tokens to a grid representation for CNN processing, which is exactly what Metta needs.

## How the Adapter Works

The `PytorchAdapter` (located in `metta/agent/external/pytorch_adapter.py`) provides seamless integration between external policies and Metta:

1. **PolicyState handling**: Automatically converts between Metta's `PolicyState` object (with `lstm_h`, `lstm_c` attributes) and PufferLib's dict format `{'lstm_h': tensor, 'lstm_c': tensor}`
2. **Smart method selection**: Chooses between `forward_eval` (for inference) and `forward` (for training) based on context
3. **State persistence**: Ensures LSTM states are properly maintained across forward passes
4. **Output format conversion**: Handles different action space formats (multi-discrete, lists of logits)

## Recommended Usage

### For Standard Token-Based Metta Environments

The unmodified `torch.py` with our `PytorchAdapter` works perfectly:
```yaml
pytorch:
  _target_: metta.agent.external.torch.Recurrent
  hidden_size: 512
```

### For Alternative Architectures

Use `lstm_transformer.py` for a hybrid LSTM-Transformer architecture:
```yaml
pytorch:
  _target_: metta.agent.external.lstm_transformer.Recurrent
  hidden_size: 384
  depth: 3
  num_heads: 6
```

The adapter system (`PytorchAdapter`) handles most interface translation automatically, making it easy to use external policies with minimal modifications.
