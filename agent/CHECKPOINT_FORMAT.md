# Metta Checkpoint Format

## Overview

Metta has transitioned from a custom pickle-based checkpoint format to standard PyTorch `.pt` files with JSON metadata sidecars. This change improves compatibility with standard PyTorch tools and eliminates brittle pickle dependencies.

## Format Specification

### New Format (v2)
Each checkpoint consists of two files:
- `{name}.pt` - Standard PyTorch state_dict containing only model weights
- `{name}.json` - JSON metadata sidecar containing all non-tensor information

Example:
```
checkpoints/
├── model_0000.pt      # Model weights (torch.save(model.state_dict()))
├── model_0000.json    # Metadata and architecture info
├── model_0001.pt
└── model_0001.json
```

### Metadata Schema

The JSON sidecar contains:

```json
{
  // Required PolicyMetadata fields
  "agent_step": 1000000,
  "epoch": 50,
  "generation": 2,
  "train_time": 3600.5,
  
  // Additional metadata
  "action_names": ["move_forward", "turn_left", "turn_right", ...],
  "run_name": "experiment_001",
  "uri": "file:///path/to/model_0050.pt",
  "avg_reward": 0.85,
  "evals": {...},
  
  // Model architecture info for reconstruction
  "model_info": {
    "type": "MettaAgent",
    "hidden_size": 256,
    "core_num_layers": 2,
    "agent_attributes": {
      "obs_shape": [34, 11, 11],
      "obs_width": 11,
      "obs_height": 11,
      "action_space": {"nvec": [9, 10]},
      "feature_normalizations": {...}
    }
  }
}
```

## Benefits

1. **Standard PyTorch Compatibility**: The `.pt` files can be loaded directly with `torch.load()` without any Metta dependencies
2. **Easy Inspection**: Metadata can be viewed/edited without loading large model files
3. **Version Control Friendly**: JSON metadata can be tracked in git
4. **Selective Loading**: Can load just metadata for analysis or just weights for inference
5. **Tool Ecosystem**: Works with standard PyTorch model analysis tools

## Loading Models

### Using PolicyStore (Recommended)
```python
from metta.agent.policy_store import PolicyStore

policy_store = PolicyStore(cfg, wandb_run)
policy_record = policy_store.policy_record("file:///path/to/model_0050.pt")
model = policy_record.policy
```

### Direct PyTorch Loading
```python
import torch
import json

# Load weights
state_dict = torch.load("model_0050.pt")

# Load metadata
with open("model_0050.json", "r") as f:
    metadata = json.load(f)

# Create model and load weights
model = create_model_from_metadata(metadata["model_info"])
model.load_state_dict(state_dict)
```

## Migration from Old Format

### Automatic Migration
Use the provided migration utility:

```bash
# Migrate single checkpoint
python -m metta.agent.migrate_checkpoints path/to/old_checkpoint.pt

# Migrate directory of checkpoints
python -m metta.agent.migrate_checkpoints path/to/checkpoint/dir --recursive

# Dry run to see what would be migrated
python -m metta.agent.migrate_checkpoints path/to/checkpoints --dry-run
```

### Manual Migration
The PolicyStore automatically handles both formats, so no action is required for existing code. Old checkpoints will be loaded transparently.

## Backward Compatibility

- PolicyStore seamlessly loads both old and new formats
- Old format: Single `.pt` file containing pickled PolicyRecord
- New format is detected by presence of `.json` sidecar
- Existing code continues to work without modification

## Future Extensions

The current dual-file format lays the groundwork for future enhancements:
- Compression options for large models
- Distributed checkpoint sharding
- Incremental checkpoint updates
- Cloud-optimized storage formats

## Implementation Details

See:
- `agent/src/metta/agent/policy_store.py` - Main save/load logic
- `agent/src/metta/agent/migrate_checkpoints.py` - Migration utility
- `agent/tests/test_policy_store.py` - Format verification tests