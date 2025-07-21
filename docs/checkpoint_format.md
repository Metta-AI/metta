# Metta Checkpoint Format

## Overview

Metta has transitioned from a custom pickle-based checkpoint format to a standard PyTorch checkpoint format with separate metadata files. This change makes checkpoints more portable, enables loading without the full Metta framework, and improves compatibility with standard PyTorch tools.

## New Format Structure

Each checkpoint now consists of two files:

1. **`model_XXXX.pt`** - Contains only the model's state_dict (PyTorch weights)
2. **`model_XXXX.json`** - Contains all metadata and model configuration

### Example Structure
```
checkpoints/
├── model_0000.pt      # PyTorch state_dict
├── model_0000.json    # Metadata sidecar
├── model_0100.pt
├── model_0100.json
└── ...
```

### Metadata JSON Schema

```json
{
  // Required fields from PolicyMetadata
  "agent_step": 1000000,
  "epoch": 100,
  "generation": 1,
  "train_time": 3600.0,
  
  // Additional metadata
  "run_name": "experiment_001",
  "uri": "file:///path/to/model_0100.pt",
  "evals": {
    "score": 0.95,
    "reward": 42.0
  },
  "avg_reward": 42.0,
  
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
      "feature_normalizations": {}
    }
  }
}
```

## Benefits

1. **Standard Format**: The .pt files contain only state_dict, making them loadable with standard `torch.load()`
2. **No Dependencies**: Models can be loaded without the Metta framework
3. **Inspectable Metadata**: JSON files can be read without loading large model files
4. **Version Control**: Metadata changes can be tracked separately from model weights
5. **Tool Compatibility**: Works with standard PyTorch model analysis tools

## Migration

### Migrating Existing Checkpoints

Use the provided migration tool to convert old checkpoints:

```bash
# Migrate a single checkpoint (creates .new.pt and .new.json)
python tools/migrate_checkpoints.py checkpoint.pt

# Migrate and replace original files
python tools/migrate_checkpoints.py checkpoint.pt --replace

# Migrate all checkpoints in a directory
python tools/migrate_checkpoints.py checkpoints/ --recursive

# Dry run to see what would be migrated
python tools/migrate_checkpoints.py checkpoints/ --dry-run
```

### Programmatic Migration

```python
from omegaconf import OmegaConf
from metta.agent.policy_store import PolicyStore

# Create PolicyStore
cfg = OmegaConf.create({
    'device': 'cpu',
    'data_dir': '/tmp',
})
policy_store = PolicyStore(cfg, wandb_run=None)

# Migrate a checkpoint
new_path = policy_store.migrate_checkpoint('old_checkpoint.pt')
```

## Loading Checkpoints

### With Metta Framework

The PolicyStore automatically detects and loads both old and new formats:

```python
# This works for both old and new formats
policy_record = policy_store.load_from_uri("file://model_0100.pt")
policy = policy_record.policy
```

### Without Metta Framework

New format checkpoints can be loaded in any PyTorch environment:

```python
import torch
import json

# Load metadata
with open('model_0100.json', 'r') as f:
    metadata = json.load(f)

# Load model weights
state_dict = torch.load('model_0100.pt')

# Reconstruct model (requires implementing model architecture)
model = YourModelClass(metadata['model_info'])
model.load_state_dict(state_dict)
```

See `tools/load_checkpoint_standalone.py` for a complete example.

## Backward Compatibility

The PolicyStore maintains full backward compatibility:

1. Automatically detects checkpoint format based on presence of .json sidecar
2. Loads old format checkpoints using the legacy loading path
3. Can migrate checkpoints on-the-fly if needed

## Future Extensions

The current implementation provides a foundation for future enhancements:

1. **Compression**: Add support for compressed .pt files
2. **Partial Loading**: Load only specific parts of large models
3. **Model Diff Storage**: Store only weight deltas for fine-tuned models
4. **Extended Metadata**: Add training curves, hyperparameter history, etc.
5. **Cloud Storage**: Direct S3/GCS support without local downloads

## Implementation Details

### Save Process

1. Extract model state_dict from PolicyRecord
2. Save state_dict to .pt file using `torch.save(model.state_dict(), path)`
3. Prepare metadata dictionary including model architecture info
4. Save metadata to .json file
5. Use atomic file operations to prevent corruption

### Load Process

1. Check for .json sidecar to determine format
2. If new format:
   - Load metadata from JSON
   - Create PolicyRecord with metadata
   - Load state_dict with `weights_only=True`
   - Reconstruct model using architecture info
   - Load weights into model
3. If old format:
   - Fall back to legacy loading code
   - Maintain full compatibility