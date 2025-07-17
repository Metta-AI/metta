# Environment Context Layer

A minimal implementation that adds environment context injection to the existing `fast.yaml` agent configuration.

## What it does

The environment context layer:
1. Takes the task name from the environment and converts it to a one-hot encoding
2. Applies a learnable context matrix C to get context vectors: C * E
3. Scales by a fixed amplitude A (currently 1.0)
4. Injects this as pre-synaptic current into the LSTM hidden state

## Usage

### Enable environment context (default)
```bash
python tools/train.py --config-name user/bullm agent=fast
```

### Disable environment context
```bash
python tools/train.py --config-name user/bullm agent=fast agent.components.env_context.enabled=false
```

### Change amplitude
```bash
python tools/train.py --config-name user/bullm agent=fast agent.components.env_context.amplitude=0.5
```

## Files Modified

1. **`configs/agent/fast.yaml`**: Added optional `env_context` layer
2. **`agent/src/metta/agent/lib/env_context.py`**: Environment context layer implementation
3. **`agent/src/metta/agent/metta_agent.py`**: Added `set_environment()` method

## Analysis

After training, you can analyze the learned context matrix by accessing the `context_matrix` parameter in the `env_context` layer of your trained model.

The context matrix has shape `[num_env_types, hidden_size]` where each row represents how that environment type is encoded in the LSTM space.
