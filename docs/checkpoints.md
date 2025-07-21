# Metta Checkpoint Format

As of January 2025, Metta uses a standard PyTorch checkpoint format for better compatibility and easier model sharing.

## New Checkpoint Format

Each checkpoint consists of two files:

1. **`{name}.pt`** - Standard PyTorch state_dict file containing only model weights
2. **`{name}.json`** - Metadata file containing all additional information

### Example Structure

```
checkpoints/
├── model_0000.pt       # Model weights (standard torch.save format)
├── model_0000.json     # Metadata (training info, architecture, etc.)
├── model_0100.pt
└── model_0100.json
```

### Benefits

- **Standard PyTorch Compatibility**: The `.pt` files can be loaded with `torch.load()` in any PyTorch environment
- **Easy Inspection**: Metadata can be viewed without loading large model files
- **Version Control**: JSON metadata files can be tracked in git
- **Partial Loading**: Load only weights or only metadata as needed
- **Tool Compatibility**: Works with standard PyTorch tools and model hubs

## Loading Checkpoints

### In Metta

```python
# Automatic format detection
policy_record = policy_store.load_from_uri("file://checkpoints/model_0100.pt")

# Access the policy
model = policy_record.policy
```

### In Pure PyTorch

```python
import torch
import json

# Load weights
state_dict = torch.load("checkpoints/model_0100.pt")

# Load metadata
with open("checkpoints/model_0100.json", "r") as f:
    metadata = json.load(f)

# Create model (using metadata for architecture info)
model = create_model_from_metadata(metadata["model_info"])
model.load_state_dict(state_dict)
```

## Metadata Structure

The JSON metadata file contains:

```json
{
  "run_name": "navigation_run_001",
  "uri": "file://checkpoints/model_0100.pt",
  "epoch": 100,
  "agent_step": 1000000,
  "generation": 1,
  "train_time": 3600.5,
  "action_names": ["move_forward", "turn_left", "turn_right"],
  "model_info": {
    "type": "MettaAgent",
    "module_name": "metta.agent.metta_agent",
    "hidden_size": 256,
    "core_num_layers": 2,
    "num_lstm_layers": 1,
    "use_lstm": true,
    "use_prev_action": true,
    "use_prev_reward": true,
    "agent_attributes": {
      "observation_space": {
        "type": "Box",
        "shape": [34, 11, 11],
        "dtype": "uint8"
      },
      "action_space": {
        "type": "MultiDiscrete",
        "nvec": [9, 10]
      }
    }
  },
  "evals": {
    "navigation": 0.85,
    "combat": 0.72
  },
  "avg_reward": 42.5
}
```

## Migration from Old Format

### Automatic Migration

Use the provided migration tool:

```bash
# Dry run to see what will be migrated
python tools/migrate_checkpoints.py checkpoints/ --dry-run

# Migrate all checkpoints in a directory
python tools/migrate_checkpoints.py checkpoints/

# Migrate a single checkpoint
python tools/migrate_checkpoints.py checkpoints/old_model.pt

# Replace original files (instead of creating .new.pt files)
python tools/migrate_checkpoints.py checkpoints/ --replace
```

### Programmatic Migration

```python
from metta.agent.policy_store import PolicyStore

# Create policy store
policy_store = PolicyStore(config, wandb_run=None)

# Migrate single checkpoint
new_path = policy_store.migrate_checkpoint("old_checkpoint.pt")

# Migrate all checkpoints in directory
from metta.agent.policy_store import migrate_all_checkpoints_in_dir
migrated = migrate_all_checkpoints_in_dir("checkpoints/", policy_store, dry_run=False)
```

## Backward Compatibility

- The PolicyStore automatically detects the checkpoint format
- Old format checkpoints can still be loaded
- Migration is recommended but not required

## Best Practices

1. **Always save both files together** - The .pt and .json files form a pair
2. **Use descriptive names** - Include epoch or timestamp in filenames
3. **Track metadata in version control** - The JSON files are small and diff-friendly
4. **Migrate old checkpoints** - Use the migration tool for better compatibility
5. **Include architecture info** - Ensure model_info in metadata has enough detail for reconstruction