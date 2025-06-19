# Metta Agents

This directory contains the refactored agent implementations that replace the old Hydra-based configuration system with direct Python classes.

## Overview

All agents inherit from `BaseAgent` and implement standard PyTorch `nn.Module` interfaces. This makes them:
- Easy to understand and debug
- Compatible with standard PyTorch tools
- Simple to extend and customize

## Available Agents

### SimpleCNNAgent
- 2 convolutional layers
- Suitable for basic tasks
- ~500K parameters

### LargeCNNAgent
- 3 convolutional layers with more channels
- Higher capacity for complex tasks
- ~2-3M parameters

### AttentionAgent
- CNN backbone with self-attention
- Better spatial reasoning
- ~1-2M parameters

### MultiHeadAttentionAgent
- Cross-attention between observations and actions
- Most sophisticated architecture
- ~3-5M parameters

## Usage

### Basic Usage

```python
from metta.agents import create_agent

# Create agent using factory
agent = create_agent(
    "simple_cnn",
    obs_space=env.observation_space,
    action_space=env.action_space,
    obs_width=env.obs_width,
    obs_height=env.obs_height,
    feature_normalizations=env.feature_normalizations,
    device="cuda"
)

# Activate actions for the environment
agent.activate_actions(
    env.action_names,
    env.max_action_args,
    device
)
```

### Direct Instantiation

```python
from metta.agents import SimpleCNNAgent

agent = SimpleCNNAgent(
    obs_space=env.observation_space,
    action_space=env.action_space,
    obs_width=11,
    obs_height=11,
    feature_normalizations={},
    hidden_size=256,  # Custom hidden size
    lstm_layers=3,    # More LSTM layers
)
```

### Creating Custom Agents

```python
from metta.agents import BaseAgent, register_agent

class MyAgent(BaseAgent):
    def __init__(self, obs_space, action_space, **kwargs):
        super().__init__(obs_space, action_space, **kwargs)

        # Your architecture here
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(128, self.hidden_size, self.lstm_layers)
        self.value_head = nn.Linear(self.hidden_size, 1)

    def compute_outputs(self, x, state):
        # Encode observations
        x = self.encoder(x)
        x = x.flatten(1)

        # LSTM processing
        x, (h, c) = self.lstm(x, (state.lstm_h, state.lstm_c))

        # Compute outputs
        value = self.value_head(x)
        logits = self.actor_head(x)  # Implement actor head

        return value, logits, (h, c)

# Register for factory use
register_agent("my_agent", MyAgent)
```

## Architecture Details

### BaseAgent

The base class provides:
- Action activation and conversion
- Standard forward pass interface
- LSTM state management
- Parameter counting

### Component Integration

Agents can use components from `metta.agent.lib`:
- `ObservationNormalizer`: Feature normalization
- `LSTM`: Recurrent processing
- `ActionEmbedding`: Action representation
- Various CNN/Linear layers

### Key Methods

- `compute_outputs()`: Main forward pass logic
- `activate_actions()`: Configure action space
- `forward()`: Standard PyTorch forward (handles training/inference)

## Migration from Old Architecture

If you have old YAML-based agent configs:

```python
from metta.util.migration import suggest_agent_migration

# Get migration suggestion
suggestion = suggest_agent_migration(old_yaml_config)
print(suggestion)
```

## Best Practices

1. **Modularity**: Break complex architectures into reusable components
2. **Type Hints**: Use type annotations for clarity
3. **Documentation**: Document custom agents thoroughly
4. **Testing**: Test agents with different observation/action spaces

## Performance Tips

- Use `torch.jit.script` for performance-critical components
- Profile with `torch.profiler` to identify bottlenecks
- Consider `torch.compile` for inference
