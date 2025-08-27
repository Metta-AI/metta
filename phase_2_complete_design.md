# Phase 2: Complete Policy System Redesign

## Executive Summary

This document outlines a **complete rewrite** of the policy saving/loading system, removing all backward compatibility and complexity identified in Phase 1. The new system uses **direct torch.save/load with weights_only=False**, eliminates all abstraction layers, and includes a **minimal metadata system** to preserve essential search/filtering capabilities.

## Phase 1 Analysis: What We're Replacing

### Current Complex Flow (TO BE DELETED)
```
TrainTool.invoke() 
→ PolicyStore.create() (493 lines - DELETED)
→ CheckpointManager.load_or_create_policy() 
  → PolicyRecord + PolicyMetadata (DELETED)
  → MettaAgent.__setstate__() - 112 lines of backward compatibility (DELETED)
→ Training Loop
→ CheckpointManager.save_policy() 
  → PolicyRecord creation with metadata (DELETED)
  → PolicyStore.save() with complex URI resolution (DELETED)
```

### Current Storage Locations (What We're Preserving)
From tracing Phase 1 code:
- **Local Storage**: `{run_dir}/checkpoints/` (from TrainTool:67, 87)
- **WandB Integration**: `upload_policy_artifact()` uploads to WandB artifacts
- **Trainer State**: Separate `trainer_state.pt` with optimizer state

### Current Metadata Analysis

#### What's Currently Stored in Metadata
From examining `checkpoint_manager.py:100-146`, the current metadata includes:

**Core Metadata (Always Saved)**
```python
metadata = {
    "epoch": epoch,                    # Training epoch number
    "agent_step": agent_step,          # Total training steps  
    "total_time": timer.get_elapsed(), # Wall clock time
    "total_train_time": rollout_time + train_time,  # Actual training time
    "run": self.run_name,              # Run identifier
    "initial_pr": initial_policy_record.uri,  # Parent checkpoint URI
}
```

**Evaluation Metadata (If Available)**
```python
metadata.update({
    "evals": {
        "category_scores": {...},       # Per-category evaluation scores  
        "simulation_scores": {...},     # Per-simulation scores
        "avg_category_score": float,    # Average category score
        "avg_simulation_score": float   # Average simulation score  
    },
    "avg_reward": avg_category_score,   # Duplicate of category score
    "score": avg_simulation_score,      # Main metric for sweeps
})
```

#### What We Care About vs. What We Can Obviate

**YES - Critical for Searching/Filtering:**
- **`score` / `avg_simulation_score`** - Used for `top` selection strategy
- **`avg_reward` / `avg_category_score`** - Alternative ranking metric
- **`epoch` / `agent_step`** - Used for `latest` selection and progress tracking
- **`run`** - Used for run identification and grouping

**MAYBE - Useful but Not Essential:**
- **`total_time`** - Nice for analysis but not critical

**NO - Can Be Obviated:**
- **`initial_pr`** - Complex parent tracking, not needed in simplified system
- **`original_feature_mapping`** - Backward compatibility, being removed
- **Complex `evals` dict** - Detailed breakdowns, can be simplified

### Policy Search Requirements
From Phase 1 audit, the system currently supports:

**Selection Strategies (PolicyStore:119-136)**
- **`top`** - Select best N by metric (score, avg_reward, etc.)
- **`latest`** - Select most recent by epoch
- **`rand`** - Random selection
- **`all`** - Return all policies

**Metric-Based Filtering (PolicyStore:182-217)**
- Extract scores from metadata: `score`, `avg_reward`, or custom metrics
- Sort by score (highest first)  
- Return top N results
- **This is the core search functionality we want to preserve**

### Versioning System
**Current**: WandB artifacts support `:v15` suffixes, local files use `model_0010.pt`
**Your Desired**: `model_0.pt` = version 0, direct epoch-to-version mapping
**Reality**: **This is already how it works locally!** The system uses `model_{epoch:04d}.pt` naming.

## New Minimal Design

### Core Principle: Direct torch.save/load + YAML Metadata
- **NO PolicyStore, NO PolicyRecord, NO PolicyMetadata objects**
- **NO URI resolution, NO selection strategies, NO caching classes**
- **NO backward compatibility whatsoever**
- **YES simple YAML sidecar files for search functionality**
- Direct `torch.save(agent, path, weights_only=False)` and `torch.load(path, weights_only=False)`

### Storage Strategy: Single Location + Optional WandB
1. **Primary**: Local storage to `{run_dir}/checkpoints/`
2. **Metadata**: YAML sidecar files for search/filtering
3. **Optional**: WandB artifact upload (simplified)
4. **Trainer State**: Separate `trainer_state.pt` for optimizer state

## Implementation Plan

### 1. New Minimal CheckpointManager

#### CheckpointManager (Simplified rewrite of existing class)
```python
class CheckpointManager:  # Rewrite existing CheckpointManager
    def __init__(self, run_dir: str, run_name: str):
        self.checkpoint_dir = f"{run_dir}/checkpoints"
        self.run_name = run_name
        
    # Core save/load methods
    def load_agent(self) -> MettaAgent | None:
        """Load agent from latest checkpoint if exists, None otherwise"""
        
    def save_agent(self, agent: MettaAgent, epoch: int, metadata: dict) -> str:
        """Save agent to model_{epoch:04d}.pt + metadata YAML, return path"""
        
    def save_trainer_state(self, optimizer: torch.optim.Optimizer, epoch: int, agent_step: int) -> None:
        """Save trainer state (optimizer + counters)"""
        
    def load_trainer_state(self) -> dict | None:
        """Load trainer state if exists"""
    
    # NEW: Metadata and search methods (preserving search functionality from Phase 1)
    def find_best_checkpoint(self, metric: str = "score") -> str | None:
        """Find checkpoint with highest score/metric"""
        
    def find_checkpoints_by_score(self, min_score: float) -> list[str]:
        """Find all checkpoints with score >= min_score"""
```

*Alternative: Develop as `SimpleCheckpointManager` during testing, then replace existing `CheckpointManager`*

### 2. Entry Points & Flow

#### Training Entry Point (Preserving run=X check)
From Phase 1 audit, this check exists in `CheckpointManager.load_or_create_policy()` lines 183-224.

**New Flow:**
```python
# In TrainTool.invoke() or equivalent:
def start_training(run_name: str, run_dir: str) -> tuple[MettaAgent, dict | None]:
    checkpoint_manager = CheckpointManager(run_dir, run_name)
    
    # Check if run=X already exists (preserving current behavior)
    existing_agent = checkpoint_manager.load_agent()
    trainer_state = checkpoint_manager.load_trainer_state()
    
    if existing_agent is not None:
        logger.info(f"Resuming existing run '{run_name}' from checkpoint")
        return existing_agent, trainer_state
    else:
        logger.info(f"Creating new run '{run_name}'")
        new_agent = MettaAgent(env, system_cfg, agent_cfg)
        return new_agent, None
```

#### Saving During Training (Preserving checkpoint intervals)
From Phase 1 audit, saving happens in `maybe_establish_checkpoint()` with intervals.

**New Flow:**
```python
def maybe_save_checkpoint(
    agent: MettaAgent, 
    optimizer: torch.optim.Optimizer,
    epoch: int, 
    agent_step: int,
    checkpoint_manager: CheckpointManager,
    interval: int,
    eval_results: dict,  # NEW: Pass evaluation results for metadata
    wandb_run: WandbRun | None = None
) -> None:
    if epoch % interval != 0:
        return
        
    # Build minimal metadata
    metadata = {
        "epoch": epoch,
        "agent_step": agent_step, 
        "score": eval_results.get("score", 0.0),
        "avg_reward": eval_results.get("avg_reward", 0.0),
        "total_time": elapsed_time,
        "run": checkpoint_manager.run_name
    }
        
    # Save agent + metadata directly
    agent_path = checkpoint_manager.save_agent(agent, epoch, metadata)
    
    # Save trainer state separately
    checkpoint_manager.save_trainer_state(optimizer, epoch, agent_step)
    
    # Optional WandB upload (simplified)
    if wandb_run:
        upload_to_wandb(wandb_run, agent_path, epoch)
```

### 3. File Structure

#### Directory Layout (Preserving current structure + metadata)
```
{run_dir}/
├── checkpoints/
│   ├── model_0000.pt      # torch.save(agent) - Direct MettaAgent
│   ├── model_0000.yaml    # epoch: 0, score: 0.75, agent_step: 1000, ...
│   ├── model_0010.pt      # Latest checkpoint (simple int sorting) 
│   ├── model_0010.yaml    # epoch: 10, score: 0.82, agent_step: 11000, ...
│   ├── model_0020.pt      # Version = epoch number (model_X.pt = version X)
│   ├── model_0020.yaml
│   └── trainer_state.pt   # torch.save({"optimizer_state": ..., "epoch": ..., "agent_step": ...})
├── logs/
└── config.json
```

#### Checkpoint Format
```python
# model_XXXX.pt - Direct agent serialization
agent = MettaAgent(...)  # Fully initialized agent
torch.save(agent, f"model_{epoch:04d}.pt", weights_only=False)

# model_XXXX.yaml - Minimal metadata for search/filtering
metadata = {
    "epoch": epoch,             # Version number (model_X.pt = version X) 
    "agent_step": agent_step,   # Training progress
    "score": eval_score,        # Main metric for search (from evaluation)
    "avg_reward": avg_reward,   # Secondary metric
    "total_time": elapsed_time, # Training duration
    "run": run_name            # Run identifier
}
import yaml
with open(f"model_{epoch:04d}.yaml", "w") as f:
    yaml.dump(metadata, f, default_flow_style=False)

# trainer_state.pt - Minimal trainer state
trainer_state = {
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
    "agent_step": agent_step
}
torch.save(trainer_state, "trainer_state.pt", weights_only=False)
```

### 4. Loading Logic

#### Latest Checkpoint Detection
```python
def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find latest model_XXXX.pt file by epoch number"""
    pattern = os.path.join(checkpoint_dir, "model_*.pt")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
        
    # Extract epoch numbers and find maximum
    epochs = []
    for f in checkpoint_files:
        match = re.match(r".*model_(\d+)\.pt", f)
        if match:
            epochs.append((int(match.group(1)), f))
    
    return max(epochs)[1] if epochs else None
```

#### Search/Filtering Implementation
```python
def find_best_checkpoint(self, metric: str = "score") -> str | None:
    """Find checkpoint with highest score/metric"""
    import yaml
    checkpoints = []
    for pt_file in glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")):
        yaml_file = pt_file.replace(".pt", ".yaml")
        if os.path.exists(yaml_file):
            with open(yaml_file) as f:
                metadata = yaml.safe_load(f)
                if metric in metadata:
                    checkpoints.append((metadata[metric], pt_file))
    
    return max(checkpoints)[1] if checkpoints else None

def find_checkpoints_by_score(self, min_score: float) -> list[str]:
    """Find all checkpoints with score >= min_score"""
    import yaml
    matching = []
    for pt_file in glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")):
        yaml_file = pt_file.replace(".pt", ".yaml")
        if os.path.exists(yaml_file):
            with open(yaml_file) as f:
                metadata = yaml.safe_load(f)
                if metadata.get("score", 0.0) >= min_score:
                    matching.append(pt_file)
    
    return sorted(matching)  # Sort by filename (epoch order)
```

## What Gets Deleted (From Phase 1 Audit)

### Complete File Deletions
- `agent/src/metta/agent/policy_store.py` (493 lines)
- `agent/src/metta/agent/policy_record.py` (252 lines)
- `agent/src/metta/agent/policy_cache.py` (77 lines)
- `agent/src/metta/agent/policy_metadata.py` (99 lines)
- `metta/rl/checkpoint_manager.py` (303 lines) - Rewrite as simplified CheckpointManager

**Total Deletion: ~1,224 lines of complex code**

### Functionality Removals
- All URI resolution (`wandb://`, `file://`, `pytorch://`)
- Policy selection strategies (`top`, `latest`, `rand`, `all`) - replaced with simple search methods
- Policy caching and LRU eviction
- Complex metadata tracking - replaced with simple YAML
- WandB artifact collection management
- All backward compatibility in `MettaAgent.__setstate__`

### Preserved Functionality
- **Run resumption**: Check if run=X exists and load latest checkpoint
- **Periodic saving**: Save at checkpoint intervals during training
- **Trainer state**: Optimizer state persistence
- **Storage location**: `{run_dir}/checkpoints/` directory structure
- **Policy search**: Find best checkpoints by score/metric (simplified)
- **Versioning**: Direct epoch-to-version mapping (`model_X.pt` = version X)
- **Optional WandB upload**: Simplified artifact upload

## Metadata System Benefits

1. **Preserves Search Functionality**: Can still find best checkpoints by score
2. **Simple Format**: YAML sidecars vs complex PolicyMetadata objects
3. **Version Alignment**: Epoch numbers = version numbers as desired
4. **Minimal Storage**: Only essential metadata, ~50 bytes per checkpoint
5. **Easy Debugging**: Human-readable YAML files
6. **Backward Compatible**: Easy to migrate existing metadata

## Implementation Steps

### Step 1: Rewrite CheckpointManager
- Replace existing 303-line CheckpointManager with simplified version
- Implement the 6 core methods shown above
- Add YAML metadata save/load functionality
- Add simple latest checkpoint detection

### Step 2: Update Training Integration
- Modify TrainTool to use simplified CheckpointManager
- Replace complex PolicyStore creation with simple agent loading
- Update training loop to pass evaluation results to save logic

### Step 3: Update Agent Creation
- Remove all PolicyRecord/PolicyStore integration from MettaAgent
- Clean up agent initialization to direct creation only
- Remove `__setstate__` backward compatibility (112 lines deleted)

### Step 4: Simplify WandB Integration
- Replace complex artifact management with direct file upload
- Remove policy artifact collections
- Keep simple model upload only

### Step 5: Clean Up Imports and References
- Remove all imports to deleted classes
- Update all references throughout codebase
- Remove unused configuration options

## Testing Strategy

### Validation Points
1. **New Run Creation**: `run=new_run` creates fresh agent and saves epoch 0
2. **Run Resumption**: `run=existing_run` loads latest checkpoint correctly  
3. **Periodic Saving**: Checkpoints saved at correct intervals with incrementing epochs
4. **Trainer State**: Optimizer state properly restored on resumption
5. **File Format**: Saved agents can be loaded with simple `torch.load()`
6. **Search Functionality**: `find_best_checkpoint()` returns highest scoring model
7. **Version Mapping**: `model_5.pt` corresponds to epoch 5, version 5

### Migration Testing
- Test that new system works with fresh runs
- **NO testing of old checkpoint compatibility** (explicitly removed)

## Success Criteria

1. **Massive Code Reduction**: Delete ~1,224 lines of complex policy management code
2. **Simple API**: 6 methods on CheckpointManager replace entire PolicyStore system
3. **Direct torch.save/load**: No abstraction layers or custom serialization
4. **Preserved Core Functionality**: Run resumption, periodic saving, and search still work
5. **Version Alignment**: `model_X.pt` = version X as requested
6. **Clean Architecture**: Single responsibility classes with clear interfaces

This design achieves the goal of "amazingly simple" by completely removing backward compatibility and complex abstractions, while preserving the core functionality needed for training workflow and adding minimal metadata support for essential search capabilities.

## Example YAML Metadata File

```yaml
# model_0010.yaml
epoch: 10
agent_step: 11000
score: 0.82
avg_reward: 0.78
total_time: 3600.5
run: "my_training_run"
```

The YAML format is more human-readable than JSON and easier for your co-worker to inspect, edit, and debug. Each metadata file is only ~6 lines and contains all the essential information needed for checkpoint search and analysis.