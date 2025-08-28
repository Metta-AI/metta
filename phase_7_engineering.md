# Phase 7 Engineering: Simplification & Goal Alignment

## Big Picture Reflection: How Are We Doing?

### Original Goal (Phase 1)
- **Target:** Simple torch.save/torch.load system
- **Key Requirement:** Delete policy saving/loading complexity, start fresh with minimal design
- **Core Functions:** 1) Check if run=X exists, 2) Load latest checkpoint with torch.load, 3) Save with torch.save at intervals

### Current Status Assessment
❌ **We've been overengineering this migration**

**What We Did (Phases 3-6):**
- Created compatibility layers (PolicyWrapper)
- Built migration bridges (SimplePolicyStore)
- Maintained eval system integration
- Preserved complex APIs for backward compatibility

**What We Should Have Done:**
- **Deleted the entire PolicyX system**
- **Built a simple 50-line checkpoint manager**
- **Updated all callers to use direct torch.save/load**

### Reality Check
- **Phase 1 Target:** ~50 lines of simple checkpoint code
- **Current State:** Still have 233-line SimpleCheckpointManager + compatibility layers
- **Original PolicyX:** 1,467 lines → **We need to get to ~50 lines total**

## Phase 7 Goal: Nuclear Simplification

**Mission:** Delete everything, implement the 3 core functions, make everything else adapt.

### Core Requirements (from original vision)

#### 1. Run Existence Check
```python
def run_exists(run_name: str) -> bool:
    """Check if run=X exists by looking for checkpoint files."""
    return (Path(f"./train_dir/{run_name}") / "checkpoints").exists()
```

#### 2. Load Latest Checkpoint  
```python
def load_latest_agent(run_name: str) -> PolicyAgent:
    """Load the latest .pt file with torch.load(weights_only=False)."""
    checkpoint_dir = Path(f"./train_dir/{run_name}/checkpoints")
    pt_files = list(checkpoint_dir.glob("*.pt"))
    if not pt_files:
        return None
    latest_file = max(pt_files, key=lambda p: p.stat().st_mtime)
    return torch.load(latest_file, weights_only=False)
```

#### 3. Save Checkpoint
```python
def save_checkpoint(agent: PolicyAgent, run_name: str, epoch: int, trainer_state: dict):
    """Save agent + trainer state with torch.save."""
    checkpoint_dir = Path(f"./train_dir/{run_name}/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(agent, checkpoint_dir / f"agent_epoch_{epoch}.pt")
    torch.save(trainer_state, checkpoint_dir / f"trainer_epoch_{epoch}.pt")
```

**That's it. That's the entire checkpoint system.**

## Current System Analysis

### What We Have Now (Over-engineered)

1. **SimpleCheckpointManager**: 233 lines doing complex metadata management
2. **PolicyWrapper/SimplePolicyStore**: Compatibility layers for eval system
3. **Complex YAML metadata files**: Storing redundant information
4. **Multi-format support**: Supporting old checkpoint formats

### What We Actually Need (Nuclear Simplification)

1. **TinyCheckpointManager**: ~50 lines total
2. **Direct torch.load/save**: No metadata files, no compatibility
3. **Eval system updated**: Work directly with checkpoint files
4. **Training integration**: 3 simple function calls

## Phase 7 Implementation Plan

### Step 1: Create TinyCheckpointManager ✨
**Target:** 50 lines maximum
```python
class TinyCheckpointManager:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.checkpoint_dir = Path(f"./train_dir/{run_name}/checkpoints")
    
    def exists(self) -> bool:
        return self.checkpoint_dir.exists()
    
    def load_latest_agent(self) -> PolicyAgent | None:
        # torch.load latest .pt file
        pass
    
    def save_checkpoint(self, agent: PolicyAgent, epoch: int, trainer_state: dict):
        # torch.save agent and trainer state
        pass
    
    def list_epochs(self) -> list[int]:
        # return available epochs by parsing filenames
        pass
```

### Step 2: Nuclear Removal 💥
**Delete Entirely:**
- `SimpleCheckpointManager` (233 lines)
- `PolicyWrapper` and `SimplePolicyStore` 
- All YAML metadata handling
- All compatibility layers
- `PolicyCache` and `PolicyMetadata`

### Step 3: Update Training Integration
**trainer.py changes:**
```python
# Replace this complex initialization:
checkpoint_manager = SimpleCheckpointManager(run_dir=..., run_name=...)

# With this:
checkpoint_manager = TinyCheckpointManager(run_name)

# Replace complex save logic with:
if epoch % save_interval == 0:
    checkpoint_manager.save_checkpoint(agent, epoch, {"optimizer": optimizer.state_dict()})

# Replace complex load logic with:
if checkpoint_manager.exists():
    agent = checkpoint_manager.load_latest_agent()
```

### Step 4: Update Eval System
**Remove PolicyRecord entirely, work directly with files:**
```python
# Old approach (complex):
policy_record = policy_store.load(uri)
scores = eval_db.get_scores(policy_record)

# New approach (simple):
agent = torch.load("./train_dir/my_run/checkpoints/agent_epoch_100.pt", weights_only=False)
scores = eval_db.get_scores("my_run", epoch=100)
```

### Step 5: Update Tools Integration
**Simplify ./tools/train.py, ./tools/sim.py etc.**
- Remove PolicyStore creation
- Remove URI parsing complexity  
- Use direct file paths

## Integration Points

### Training Pipeline
**Current (Complex):**
```
TrainTool → PolicyStore → SimpleCheckpointManager → Complex metadata → YAML files
```

**Target (Simple):**
```
TrainTool → TinyCheckpointManager → torch.save/load → Done
```

### Evaluation Pipeline  
**Current (Complex):**
```
SimTool → PolicyStore → PolicyRecord → Eval system compatibility layers
```

**Target (Simple):**
```
SimTool → torch.load(checkpoint_file) → Direct eval with agent object
```

### Database Integration
**Current (Complex):**
```
PolicyRecord metadata → Complex key/version mapping → Database queries
```

**Target (Simple):**
```
Checkpoint filename parsing → Direct epoch/run extraction → Database queries
```

## File Deletion Plan

### Phase 7A: Delete Complex Components
```bash
rm metta/rl/simple_checkpoint_manager.py        # 233 lines gone
rm metta/sim/simple_policy_store.py             # 55 lines gone  
rm agent/src/metta/agent/policy_metadata.py     # Metadata handling gone
rm agent/src/metta/agent/policy_cache.py        # Caching complexity gone
```

### Phase 7B: Create Minimal Replacement
```bash
# Create single file:
metta/rl/tiny_checkpoint_manager.py             # ~50 lines total
```

### Phase 7C: Update Integration Points
- Update `trainer.py` to use TinyCheckpointManager
- Update `tools/sim.py` to use direct torch.load
- Update eval system to work with checkpoint files directly
- Remove all PolicyStore/PolicyRecord references

## Success Criteria

### Quantitative Goals
- **Lines of Code:** Reduce checkpoint system from 233 lines to ~50 lines
- **File Count:** Reduce checkpoint-related files from 5+ to 1
- **Test Complexity:** Simplify checkpoint tests to basic torch.save/load validation

### Functional Goals  
- ✅ Training can save/load checkpoints
- ✅ Evaluation works with saved checkpoints
- ✅ Tools can find and use existing runs
- ✅ No backward compatibility burden

### Integration Goals
- ✅ PolicyEvaluator works with direct checkpoint loading
- ✅ Database integration uses checkpoint filename parsing  
- ✅ All tools work with simplified checkpoint system

## Phase 7 Timeline

### Week 1: Nuclear Deletion
- Delete SimpleCheckpointManager and compatibility layers
- Create TinyCheckpointManager (50 lines)
- Update trainer.py integration

### Week 2: Integration Updates  
- Update all tools (sim.py, train.py, etc.)
- Update eval system to work directly with checkpoint files
- Update database integration

### Week 3: Testing & Validation
- Validate end-to-end training → evaluation pipeline
- Ensure PolicyEvaluator integration works
- Clean up any remaining complexity

## Expected Challenges

1. **Test Failures:** Many tests will break when we delete compatibility layers
   - **Solution:** Update tests to use direct torch.load/save patterns

2. **Eval System Integration:** Database queries currently expect PolicyRecord metadata
   - **Solution:** Parse epoch/run info directly from checkpoint filenames

3. **Tool Integration:** Tools currently use complex URI parsing  
   - **Solution:** Simplify to direct file path handling

## The Big Win

After Phase 7, we'll have achieved the original vision:
- **Simple checkpoint system:** ~50 lines instead of 1,467 lines
- **Direct torch.save/load:** No complex abstraction layers
- **Clean integration:** Tools work directly with checkpoint files  
- **Zero backward compatibility:** Fresh start with modern PyTorch patterns

This is the simplification we should have done from the beginning, and Phase 7 is our opportunity to get back on track with the original goal.