# Agent States Recorded in Repository

## Overview

**Yes, agent states are recorded extensively** across multiple systems in this repository. The codebase implements comprehensive state tracking for different types of agents: simulation agents, AI assistant bots, neural network agents, and replay systems.

## 1. MettaGrid Simulation Agents (C++)

Agent states are fully tracked in the simulation environment with the following components:

### Location and Movement Tracking
- **Position**: Grid coordinates, layer information
- **Orientation**: Agent facing direction
- **Movement History**: Previous location tracking
- **Visitation Grids**: Count of visits to each grid position

### Inventory Management
- **Items**: Map of item types to quantities
- **Resource Limits**: Maximum quantities per item type
- **Resource Rewards**: Reward values for collecting items
- **Stat Rewards**: Performance-based rewards

### Behavioral State
- **Frozen Status**: Whether agent is temporarily disabled
- **Freeze Duration**: How long the agent remains frozen
- **Action Failure Penalties**: Costs for failed actions
- **Motion Tracking**: Steps without movement

### Performance Tracking
- **Stats Tracker**: Comprehensive performance metrics
- **Current Rewards**: Immediate reward values
- **Total Rewards**: Cumulative reward accumulation

**Location**: `mettagrid/src/metta/mettagrid/objects/agent.hpp`

```cpp
std::map<InventoryItem, InventoryQuantity> inventory;
std::vector<std::vector<unsigned int>> visitation_grid;
RewardType current_stat_reward;
StatsTracker stats;
ObservationType group;
short frozen;
short freeze_duration;
Orientation orientation;
```

## 2. CodeBot Agents (Python)

Bot states are persistently recorded with detailed work cycle tracking and lifecycle management.

### Bot Lifecycle
- **Identity**: Unique bot ID and name
- **Status**: Active, paused, or terminated states
- **Timestamps**: Creation time and last activity

### Work Tracking
- **Goals Completed**: Number of completed objectives
- **Current Goal**: Active goal being pursued
- **Work Cycle History**: List of completed work cycles

### Relationships
- **Parent/Child Hierarchy**: Bot organizational structure
- **Delegated Responsibilities**: File paths and ownership assignments
- **Blocked Tasks**: Tasks waiting for resolution

### Work Cycle Records
- **Cycle ID**: Unique identifier for each work session
- **Timestamp**: When the work cycle occurred
- **Goal Progress**: Percentage completion of objectives
- **Files Changed**: List of modified files
- **Commands Executed**: Terminal commands run
- **Outcome**: Success, partial, or blocked status
- **Blockers**: Issues preventing completion

**Location**: `codebot/README.md`

```python
class BotState(BaseModel):
    """Persistent state for a manybot"""
    bot_id: str
    name: str
    status: Literal["active", "paused", "terminated"]
    created_at: datetime
    last_active: datetime

    # Work tracking
    goals_completed: int = 0
    current_goal: Optional[Goal] = None
    work_cycles: List[WorkCycleRecord] = Field(default_factory=list)

    # Relationships
    parent_bot: Optional[str] = None
    child_bots: List[str] = Field(default_factory=list)
    delegated_from: Dict[str, List[str]] = Field(default_factory=dict)

class WorkCycleRecord(BaseModel):
    """Record of a single work cycle"""
    cycle_id: str
    timestamp: datetime
    goal_progress: float
    files_changed: List[str]
    commands_executed: List[str]
    outcome: Literal["success", "partial", "blocked"]
    blockers: List[str] = Field(default_factory=list)
```

## 3. PyTorch Neural Network Agents

LSTM-based agents have sophisticated memory state management with per-environment tracking.

### Memory State Management
- **LSTM Hidden States**: `lstm_h` - Hidden layer activations
- **LSTM Cell States**: `lstm_c` - Long-term memory storage
- **Per-Environment Tracking**: Separate states for each training environment
- **Episode Boundary Reset**: Automatic state reset at episode ends

### Memory Management Features
- **Gradient Detachment**: Prevents memory leaks during training
- **State Initialization**: Proper initialization of new environments
- **Memory Reset**: Manual and automatic state clearing
- **Checkpoint Support**: Full save/load capability

### Key Methods
- `has_memory()`: Indicates presence of LSTM states
- `get_memory()`: Retrieves states for checkpointing
- `set_memory()`: Restores states from checkpoint
- `reset_memory()`: Clears all states
- `reset_env_memory()`: Clears specific environment states

**Location**: `agent/src/metta/agent/pytorch/base.py`

```python
def get_memory(self):
    """Get current LSTM memory states for checkpointing."""
    return self.lstm_h, self.lstm_c

def set_memory(self, memory):
    """Set LSTM memory states from checkpoint."""
    self.lstm_h, self.lstm_c = memory[0], memory[1]

def reset_memory(self):
    """Reset all LSTM memory states."""
    self.lstm_h.clear()
    self.lstm_c.clear()
```

## 4. Replay System (Nim/TypeScript)

Complete agent state histories are recorded for analysis and visualization.

### Temporal State Tracking
- **Action Sequences**: All actions taken by agents
- **Action Parameters**: Specific parameters for each action
- **Action Success Rates**: Boolean success indicators
- **Reward History**: Reward values over time

### Agent Properties Over Time
- **Position History**: Location changes throughout simulation
- **Orientation Changes**: Facing direction evolution
- **Inventory Evolution**: Item quantities over time
- **Frozen State Tracking**: Freeze status and progress

### Building State Tracking
- **Resource Management**: Input/output resource tracking
- **Production Progress**: Manufacturing completion status
- **Cooldown Management**: Recovery time tracking

**Location**: `mettascope2/src/mettascope/replays.nim`

```nim
type
  Entity* = ref object
    # Common keys.
    id*: int
    typeId*: int
    groupId*: int
    agentId*: int
    location*: seq[IVec3]
    orientation*: seq[int]
    inventory*: seq[seq[ItemAmount]]
    inventoryMax*: int
    color*: seq[int]

    # Agent specific keys.
    actionId*: seq[int]
    actionParameter*: seq[int]
    actionSuccess*: seq[bool]
    currentReward*: seq[float]
    totalReward*: seq[float]
    isFrozen*: seq[bool]
    frozenProgress*: seq[int]
    frozenTime*: int
    visionSize*: int
```

## 5. Policy Checkpoint System

Neural network agent states are saved and restored through a sophisticated checkpoint system.

### Checkpoint Features
- **Model State Preservation**: Complete neural network state
- **Optimizer State Tracking**: Training optimizer parameters
- **Metadata Management**: Version and configuration tracking
- **Artifact Management**: Policy storage and retrieval

### Key Components
- **PolicyRecord**: Container for policy and metadata
- **PolicyMetadata**: Configuration and version information
- **PolicyStore**: Storage and retrieval system

**Location**: `agent/src/metta/agent/policy_record.py`

## 6. Checkpoint Specialist Documentation

Comprehensive documentation exists for checkpoint validation and management.

### Expertise Areas
- PyTorch checkpoint serialization and state_dict management
- Optimizer state preservation and restoration
- Distributed model checkpoint coordination
- Model versioning and migration strategies
- Checkpoint compression and storage optimization

**Location**: `.claude/agents/checkpoint.md`

## Summary

The repository implements **comprehensive agent state recording** across four distinct domains:

1. **Simulation Agents**: Full state tracking in grid-based environments
2. **AI Assistant Bots**: Persistent state with work cycle history
3. **Neural Network Agents**: LSTM memory state management
4. **Replay Analysis**: Temporal state evolution tracking

Each system is optimized for its specific use case while maintaining robust state persistence, recovery, and analysis capabilities.
