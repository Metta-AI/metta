# Phase 2: Minimal Policy System Redesign

## Executive Summary

This document outlines a **complete rewrite** of the policy saving/loading system, removing all backward compatibility and complexity identified in Phase 1. The new system uses **direct torch.save/load with weights_only=False** and eliminates all abstraction layers.

## Phase 1 Flow Analysis (What We're Replacing)

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

## New Minimal Design

### Core Principle: Direct torch.save/load Only
- **NO PolicyStore, NO PolicyRecord, NO PolicyMetadata**
- **NO URI resolution, NO selection strategies, NO caching**
- **NO backward compatibility whatsoever**
- Direct `torch.save(agent, path, weights_only=False)` and `torch.load(path, weights_only=False)`

### Storage Strategy: Single Location + Optional WandB
Based on current system analysis, we preserve:
1. **Primary**: Local storage to `{run_dir}/checkpoints/`
2. **Optional**: WandB artifact upload (simplified)
3. **Trainer State**: Separate `trainer_state.pt` for optimizer state

## Implementation Plan

### 1. New Minimal Classes

#### SimpleCheckpointManager (Replaces all of Phase 1 complexity)
```python
class SimpleCheckpointManager:
    def __init__(self, run_dir: str, run_name: str):
        self.checkpoint_dir = f"{run_dir}/checkpoints"
        self.run_name = run_name
        
    def load_latest_agent(self) -> MettaAgent | None:
        """Load latest checkpoint if exists, None otherwise"""
        
    def save_agent(self, agent: MettaAgent, epoch: int) -> str:
        """Save agent to model_{epoch:04d}.pt, return path"""
        
    def save_trainer_state(self, optimizer: torch.optim.Optimizer, epoch: int, agent_step: int) -> None:
        """Save trainer state (optimizer + counters)"""
        
    def load_trainer_state(self) -> dict | None:
        """Load trainer state if exists"""
```

### 2. Entry Points & Flow

#### Training Entry Point (Preserving run=X check)
From Phase 1 audit, this check exists in `CheckpointManager.load_or_create_policy()` lines 183-224.

**New Flow:**
```python
# In TrainTool.invoke() or equivalent:
def start_training(run_name: str, run_dir: str) -> tuple[MettaAgent, dict | None]:
    checkpoint_manager = SimpleCheckpointManager(run_dir, run_name)
    
    # Check if run=X already exists (preserving current behavior)
    existing_agent = checkpoint_manager.load_latest_agent()
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
    checkpoint_manager: SimpleCheckpointManager,
    interval: int,
    wandb_run: WandbRun | None = None
) -> None:
    if epoch % interval != 0:
        return
        
    # Save agent directly
    agent_path = checkpoint_manager.save_agent(agent, epoch)
    
    # Save trainer state separately
    checkpoint_manager.save_trainer_state(optimizer, epoch, agent_step)
    
    # Optional WandB upload (simplified)
    if wandb_run:
        upload_to_wandb(wandb_run, agent_path, epoch)
```

### 3. File Structure

#### Directory Layout (Preserving current structure)
```
{run_dir}/
├── checkpoints/
│   ├── model_0000.pt      # torch.save(agent) - Direct MettaAgent
│   ├── model_0010.pt      # Latest checkpoint (simple int sorting)
│   ├── model_0020.pt
│   └── trainer_state.pt   # torch.save({"optimizer_state": ..., "epoch": ..., "agent_step": ...})
├── logs/
└── config.json
```

#### Checkpoint Format
```python
# model_XXXX.pt - Direct agent serialization
agent = MettaAgent(...)  # Fully initialized agent
torch.save(agent, f"model_{epoch:04d}.pt", weights_only=False)

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

## What Gets Deleted (From Phase 1 Audit)

### Complete File Deletions
- `agent/src/metta/agent/policy_store.py` (493 lines)
- `agent/src/metta/agent/policy_record.py` (252 lines)
- `agent/src/metta/agent/policy_cache.py` (77 lines)
- `agent/src/metta/agent/policy_metadata.py` (99 lines)
- `metta/rl/checkpoint_manager.py` (303 lines) - Replace with SimpleCheckpointManager

**Total Deletion: ~1,224 lines of complex code**

### Functionality Removals
- All URI resolution (`wandb://`, `file://`, `pytorch://`)
- Policy selection strategies (`top`, `latest`, `rand`, `all`)
- Policy caching and LRU eviction
- Complex metadata tracking
- WandB artifact collection management
- All backward compatibility in `MettaAgent.__setstate__`

### Preserved Functionality
- **Run resumption**: Check if run=X exists and load latest checkpoint
- **Periodic saving**: Save at checkpoint intervals during training
- **Trainer state**: Optimizer state persistence
- **Storage location**: `{run_dir}/checkpoints/` directory structure
- **Optional WandB upload**: Simplified artifact upload

## Implementation Steps

### Step 1: Create SimpleCheckpointManager
- Implement the 4 core methods shown above
- Add basic file I/O with atomic writes
- Add simple latest checkpoint detection

### Step 2: Replace Training Integration
- Modify TrainTool to use SimpleCheckpointManager
- Replace complex PolicyStore creation with simple agent loading
- Update training loop to use simple save logic

### Step 3: Update Agent Creation
- Remove all PolicyRecord/PolicyStore integration from MettaAgent
- Clean up agent initialization to direct creation only
- Remove `__setstate__` backward compatibility

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

### Migration Testing
- Test that new system works with fresh runs
- **NO testing of old checkpoint compatibility** (explicitly removed)

## Success Criteria

1. **Massive Code Reduction**: Delete ~1,224 lines of complex policy management code
2. **Simple API**: 4 methods on SimpleCheckpointManager replace entire PolicyStore system
3. **Direct torch.save/load**: No abstraction layers or custom serialization
4. **Preserved Core Functionality**: Run resumption and periodic saving still work
5. **Clean Architecture**: Single responsibility classes with clear interfaces

This design achieves the goal of "amazingly simple" by completely removing backward compatibility and complex abstractions, while preserving the core functionality needed for training workflow.