# MettaAgent Saving and Loading Guide

This guide explains how to save and load MettaAgent checkpoints with the unified save/load API.

## Overview

MettaAgent provides a single save/load interface with an optional `full_model` parameter:

| Method | full_model | Use Case | File Size | Load Speed | Reconstruction Required |
|--------|------------|----------|-----------|------------|------------------------|
| `save()` / `load()` | False (default) | Evaluation, sharing | Small | Slower | Yes |
| `save()` / `load()` | True | Training resumption | Large | Fast | No |

## Methods

### 1. Standard Save/Load (Default - for Evaluation)

```python
# Saving
agent = MettaAgent(model=brain_policy, model_type="brain")
agent.save("checkpoint.pt")  # full_model=False by default

# Loading
loaded_agent = MettaAgent.load("checkpoint.pt", device="cuda")
```

**What's saved:**
- Model state dictionary (weights only)
- Model type and configuration
- Metadata (epoch, scores, etc.)
- Agent attributes for reconstruction

**Pros:**
- Smaller file size (only weights)
- Portable across similar code versions
- Good for model sharing and evaluation

**Cons:**
- Requires model reconstruction
- May fail if model architecture changed significantly
- Slower loading time

### 2. Full Model Save/Load (for Training)

```python
# Saving
agent.save("training_checkpoint.pt", full_model=True)

# Loading
agent = MettaAgent.load("training_checkpoint.pt", device="cuda", full_model=True)
```

**What's saved:**
- Complete model object (pickled)
- All metadata
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

## Best Practices

### 1. For Model Sharing/Evaluation

Use standard save (default):
```python
# Include comprehensive metadata
agent.metadata.update({
    "training_config": cfg,
    "final_scores": eval_scores,
    "training_time": total_time,
    "git_hash": get_git_hash(),
})
agent.save("final_model.pt")  # full_model=False by default
```

### 2. For Training Checkpoints

Use full model save with periodic cleanup:
```python
# Save periodically during training
if epoch % checkpoint_interval == 0:
    agent.save(f"checkpoint_epoch_{epoch}.pt", full_model=True)

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

    agent.save("checkpoint.pt", full_model=True)
```

### 4. For Production Deployment

Create a minimal checkpoint:
```python
# Save only essential data for inference
agent.save("production_model.pt")  # Default full_model=False is perfect for this
```

## Troubleshooting

### Common Issues

1. **"No module named 'agent'"**
   - Old checkpoint with legacy module paths
   - Solution: Load with PolicyStore (handles compatibility)

2. **"Model reconstruction failed"**
   - Missing model configuration in checkpoint
   - Solution: Use full_model=True or ensure matching code version

3. **Large checkpoint files**
   - Using full_model=True for evaluation
   - Solution: Re-save with full_model=False

### Loading Unknown Checkpoints

```python
# Safe loading pattern
try:
    # Try with full_model=True first (faster if it works)
    agent = MettaAgent.load(path, device=device, full_model=True)
except Exception as e:
    logger.warning(f"Full model load failed: {e}")
    try:
        # Try standard method
        agent = MettaAgent.load(path, device=device, full_model=False)
    except Exception as e:
        logger.warning(f"Standard load failed: {e}")
        # Last resort: use PolicyStore
        policy_store = PolicyStore(cfg, None)
        agent = policy_store.policy(f"file://{path}")
```

## File Format Details

### Standard Save Format (full_model=False)
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
    "agent_attributes": {...},  # For BrainPolicy reconstruction
}
```

### Full Model Save Format (full_model=True)
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
- Full model save: 50-500 MB (includes full objects)

### Loading Times
- Standard load: 1-5 seconds (includes reconstruction)
- Full model load: 0.1-0.5 seconds (direct unpickling)
- From wandb: +5-30 seconds (download time)

### Memory Usage
- Keep only one agent in memory during evaluation
- Use `del agent` and `torch.cuda.empty_cache()` when switching models
- Consider memory-mapped loading for very large models

## Loading Legacy Checkpoints

The system automatically handles various legacy formats:
- Old MettaAgent checkpoints
- Raw state dictionaries
- Raw model objects

The PolicyStore provides the most robust loading with automatic fallbacks:
```python
policy_store = PolicyStore(cfg, None)
agent = policy_store.load_from_uri("file:///path/to/any_checkpoint.pt")
```
