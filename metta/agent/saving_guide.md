# MettaAgent Saving and Loading Guide

This guide explains the different methods for saving and loading MettaAgent checkpoints, when to use each method, and their trade-offs.

## Overview

MettaAgent provides multiple save/load methods optimized for different use cases:

| Method | Use Case | File Size | Load Speed | Reconstruction Required |
|--------|----------|-----------|------------|------------------------|
| `save()` / `load()` | Evaluation, sharing | Small | Slower | Yes |
| `save_for_training()` / `load_for_training()` | Training resumption | Large | Fast | No |
| `torch.save()` (raw) | Custom use cases | Variable | Variable | Yes |

## Methods

### 1. Standard Save/Load (Recommended for Evaluation)

```python
# Saving
agent = MettaAgent(model=brain_policy, model_type="brain")
agent.save("checkpoint.pt")

# Loading
loaded_agent = MettaAgent.load("checkpoint.pt", device="cuda")
```

**What's saved:**
- Model state dictionary (weights only)
- Model type and configuration
- Metadata (epoch, scores, etc.)
- Version information
- Agent attributes for reconstruction

**Pros:**
- Smaller file size (only weights)
- Portable across code versions
- Can be migrated to new formats

**Cons:**
- Requires model reconstruction
- May fail if model architecture changed significantly
- Slower loading time

### 2. Training Save/Load (Recommended for Training)

```python
# Saving
agent.save_for_training("training_checkpoint.pt")

# Loading
agent = MettaAgent.load_for_training("training_checkpoint.pt", device="cuda")
```

**What's saved:**
- Complete model object (pickled)
- All metadata and version info
- Ready to resume training immediately

**Pros:**
- Fast loading (no reconstruction)
- Preserves exact model state
- Includes all custom attributes

**Cons:**
- Large file size
- Not portable across code versions
- Can break if code structure changes

### 3. PolicyStore Methods (Integrated Workflow)

```python
# Through PolicyStore (handles paths and metadata)
policy_store = PolicyStore(cfg, wandb_run)

# Save with automatic naming and metadata
agent = policy_store.save(
    name="model_0100.pt",
    path="/path/to/checkpoints/model_0100.pt",
    agent=agent,
    metadata={"epoch": 100, "score": 0.95}
)

# Load with various options
agent = policy_store.policy("file:///path/to/checkpoint.pt")
agents = policy_store.policies("file:///path/to/checkpoints/", n=5, metric="score")
```

**Features:**
- Automatic caching
- Metadata management
- Support for wandb artifacts
- Selection strategies (top, latest, random)

## Version Compatibility

### Checkpoint Format Versions

- **Version 1**: Legacy format (pre-refactor)
- **Version 2**: Current format with version tracking

### Checking Compatibility

```python
from metta.agent.version_compatibility import check_checkpoint_compatibility

# Check before loading
report = check_checkpoint_compatibility("checkpoint.pt")
print(report)
# Output:
# Compatibility Level: full
# Warnings:
#   - Action space version mismatch...
```

### Migration Tool

For old checkpoints:

```bash
# Migrate single file
python -m metta.agent.migrate_checkpoints old_checkpoint.pt

# Migrate directory
python -m metta.agent.migrate_checkpoints checkpoints/ migrated/

# Without backup
python -m metta.agent.migrate_checkpoints checkpoint.pt --no-backup
```

## Best Practices

### 1. For Model Sharing/Evaluation

Use standard `save()`:
```python
# Include comprehensive metadata
agent.metadata.update({
    "training_config": cfg,
    "final_scores": eval_scores,
    "training_time": total_time,
    "git_hash": get_git_hash(),
})
agent.save("final_model.pt")
```

### 2. For Training Checkpoints

Use `save_for_training()` with periodic cleanup:
```python
# Save periodically during training
if epoch % checkpoint_interval == 0:
    agent.save_for_training(f"checkpoint_epoch_{epoch}.pt")

    # Clean up old checkpoints to save space
    if epoch > keep_last_n_checkpoints * checkpoint_interval:
        old_epoch = epoch - keep_last_n_checkpoints * checkpoint_interval
        os.remove(f"checkpoint_epoch_{old_epoch}.pt")
```

### 3. For Distributed Training

Ensure only rank 0 saves:
```python
if torch.distributed.get_rank() == 0:
    # Unwrap from DistributedDataParallel if needed
    if isinstance(agent, DistributedMettaAgent):
        agent = agent.module

    agent.save("checkpoint.pt")
```

### 4. For Production Deployment

Create a minimal checkpoint:
```python
# Save only essential data for inference
checkpoint = {
    "model_state_dict": agent.state_dict(),
    "action_names": agent.metadata["action_names"],
    "model_config": minimal_config,
}
torch.save(checkpoint, "production_model.pt")
```

## Troubleshooting

### Common Issues

1. **"No module named 'agent'"**
   - Old checkpoint with legacy module paths
   - Solution: Load with PolicyStore (handles compatibility)

2. **"Model reconstruction failed"**
   - Missing model configuration in checkpoint
   - Solution: Use migration tool or load_for_training

3. **"Checkpoint format version X not supported"**
   - Newer checkpoint with older code
   - Solution: Update code or migrate checkpoint

4. **Large checkpoint files**
   - Using save_for_training for evaluation
   - Solution: Re-save with standard save() method

### Loading Unknown Checkpoints

```python
# Safe loading pattern
try:
    # Try fast method first
    agent = MettaAgent.load_for_training(path)
except Exception as e:
    logger.warning(f"Fast load failed: {e}")
    try:
        # Try standard method
        agent = MettaAgent.load(path)
    except Exception as e:
        logger.warning(f"Standard load failed: {e}")
        # Last resort: use PolicyStore
        policy_store = PolicyStore(cfg, None)
        agent = policy_store.policy(f"file://{path}")
```

## File Format Details

### Standard Save Format (v2)
```python
{
    "checkpoint_format_version": 2,
    "model_state_dict": OrderedDict(...),  # PyTorch state dict
    "model_type": "brain",  # or "pytorch"
    "name": "model_0100.pt",
    "uri": "file:///path/to/model_0100.pt",
    "metadata": {
        "epoch": 100,
        "agent_step": 1000000,
        "score": 0.95,
        # ... other metadata
    },
    "observation_space_version": "v1",
    "action_space_version": "v1",
    "layer_version": "v1",
    "agent_attributes": {...},  # For BrainPolicy reconstruction
}
```

### Training Save Format
```python
{
    "checkpoint_format_version": 2,
    "model": <complete model object>,  # Pickled nn.Module
    "model_type": "brain",
    # ... same metadata as standard format
}
```

## Performance Considerations

### File Sizes (Approximate)
- Standard save: 5-50 MB (weights only)
- Training save: 50-500 MB (includes optimizer state, full objects)
- Compressed: 30-70% reduction with `torch.save(..., compress=True)`

### Loading Times
- Standard load: 1-5 seconds (includes reconstruction)
- Training load: 0.1-0.5 seconds (direct unpickling)
- From wandb: +5-30 seconds (download time)

### Memory Usage
- Keep only one agent in memory during evaluation
- Use `del agent` and `torch.cuda.empty_cache()` when switching models
- Consider memory-mapped loading for very large models

## Future Improvements

Planned enhancements:
1. Incremental checkpointing (save only deltas)
2. Automatic compression for large checkpoints
3. Cloud-native checkpoint streaming
4. Version auto-migration on load
5. Checkpoint validation and repair tools
