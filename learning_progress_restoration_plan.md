# Learning Progress Restoration Plan

## Overall Goal

Restore learning progress functionality as it existed pre-dehydration by implementing both the **simple learning progress** (minimal variant) and **full learning progress** (BidirectionalLearningProgress) from the original working system. These will be integrated into the current CogWorks curriculum system as multiple variants for testing and comparison.

## Pre-Dehydration State Analysis

From our comprehensive audit at commit `4167f6684`, we found two working learning progress implementations:

### 1. Simple Learning Progress (`learning_progress_minimal.py`)
- **Architecture**: Direct EMA tracking per task instance
- **Performance**: O(n) task management but simple and reliable
- **Features**: Fast/slow EMA, bounded memory, basic learning progress calculation
- **Integration**: Standalone curriculum with basic task management

### 2. Sophisticated Learning Progress (`mettagrid/learning_progress.py`)
- **Architecture**: BidirectionalLearningProgress with fixed task indexing
- **Performance**: O(1) task indexing, sophisticated sampling algorithms
- **Features**: Progress smoothing, sigmoid normalization, random baseline, memory management
- **Integration**: Minimal override of RandomCurriculum.complete_task()

## Implementation Strategy

### Phase 1: Simple Learning Progress Restoration
**Goal**: Port the working minimal learning progress to current CogWorks system

#### 1.1 Create SimpleLearningProgressAlgorithm
```python
# File: metta/cogworks/curriculum/simple_learning_progress.py
class SimpleLearningProgressConfig(CurriculumAlgorithmConfig):
    ema_timescale: float = 0.001
    memory: int = 25

class SimpleLearningProgressAlgorithm(CurriculumAlgorithm):
    # Direct port of learning_progress_minimal.py logic
    # Per-task EMA tracking with _p_fast/_p_slow
    # Learning progress = abs(_p_fast - _p_slow)
```

#### 1.2 Integration Points
- **Task Tracking**: Use existing CurriculumTask with per-task outcome tracking
- **Scoring**: Override `score_tasks()` to return learning progress scores
- **Memory Management**: Bounded outcome lists per task (default 25)
- **Configuration**: Pydantic config integrated with CurriculumConfig

### Phase 2: BidirectionalLearningProgress Port
**Goal**: Port the sophisticated MettagGrid learning progress algorithm

#### 2.1 Create BidirectionalLearningProgress Module
```python
# File: metta/cogworks/curriculum/bidirectional_learning_progress.py
class BidirectionalLearningProgressConfig(CurriculumAlgorithmConfig):
    ema_timescale: float = 0.001
    progress_smoothing: float = 0.05
    num_active_tasks: int = 16
    rand_task_rate: float = 0.25
    sample_threshold: int = 10
    memory: int = 25

class BidirectionalLearningProgressAlgorithm(CurriculumAlgorithm):
    # Port BidirectionalLearningProgress class
    # Task type enumeration for O(1) performance
    # Full sigmoid normalization and progress smoothing
```

#### 2.2 Task Type Mapping Strategy
Since the original used fixed task indices, we'll create a mapping system:

1. **Task Type Enumeration**: Use `TaskGenerator.get_all_task_types()` to get fixed task types
2. **Task Type Indexing**: Map task types to indices (0 to n-1) for O(1) lookup
3. **Task Instance → Type Mapping**: Extract task type from CurriculumTask bucket values
4. **BidirectionalLearningProgress Integration**: Use task type indices instead of task IDs

### Phase 3: Navigation Recipe Variants
**Goal**: Create multiple navigation training recipes to test all variants

#### 3.1 Navigation Recipe Functions
```python
# In experiments/recipes/navigation.py

def train_simple_lp(**kwargs):
    """Navigation training with simple learning progress."""
    config = make_navigation_curriculum_config()
    config.algorithm_config = SimpleLearningProgressConfig()
    return make_navigation_tool(config, **kwargs)

def train_bidirectional_lp(**kwargs):
    """Navigation training with bidirectional learning progress.""" 
    config = make_navigation_curriculum_config()
    config.algorithm_config = BidirectionalLearningProgressConfig()
    return make_navigation_tool(config, **kwargs)

def train_current_lp(**kwargs):
    """Navigation training with current (enhanced) learning progress."""
    config = make_navigation_curriculum_config() 
    config.algorithm_config = LearningProgressConfig()  # Current system
    return make_navigation_tool(config, **kwargs)

def train_baseline(**kwargs):
    """Navigation training with random curriculum (baseline)."""
    config = make_navigation_curriculum_config()
    config.algorithm_config = DiscreteRandomConfig()
    return make_navigation_tool(config, **kwargs)
```

#### 3.2 Testing Commands
```bash
# Test simple learning progress
uv run ./tools/run.py experiments.recipes.navigation.train_simple_lp run=simple_lp_test

# Test bidirectional learning progress  
uv run ./tools/run.py experiments.recipes.navigation.train_bidirectional_lp run=bidirectional_lp_test

# Test current learning progress
uv run ./tools/run.py experiments.recipes.navigation.train_current_lp run=current_lp_test

# Test baseline random
uv run ./tools/run.py experiments.recipes.navigation.train_baseline run=baseline_test
```

## Technical Implementation Details

### Task Type Integration Strategy

#### Current Task Generator Enhancement
We've already added task type enumeration to TaskGenerator:
```python
def get_all_task_types(self) -> list[str]:
    """Get all possible task types this generator can produce."""
    
def get_task_by_type(self, task_type: str, task_id: int) -> MettaGridConfig:
    """Generate a task of a specific type."""
```

#### BucketedTaskGenerator Implementation
The BucketedTaskGenerator creates 17 task types for navigation:
- Map-based types: `map_terrain`, `map_nohearts`, etc.
- Object density types: `sparse_objects`, `medium_objects`, `dense_objects`

#### Task Type Mapping
```python
class TaskTypeMapper:
    """Maps between task instances and task type indices."""
    
    def __init__(self, task_generator):
        self._task_types = task_generator.get_all_task_types()
        self._type_to_index = {t: i for i, t in enumerate(self._task_types)}
    
    def get_task_type_index(self, task: CurriculumTask) -> int:
        """Get task type index from task instance."""
        # Extract task type from bucket values or task properties
        # Return corresponding index (0 to n-1)
```

### Performance Characteristics

| Variant | Task Operations | Memory Usage | Complexity | Use Case |
|---------|----------------|--------------|------------|----------|
| Simple LP | O(n) per task | Bounded per task | O(n) | Reliable, debuggable |
| Bidirectional LP | O(1) per type | Fixed arrays | O(k) where k=types | Maximum performance |
| Current LP | O(n) per task | Bounded + caching | O(n) | Feature-rich |

### Configuration Integration

All variants integrate with the existing configuration system:

```python
# In CurriculumConfig
algorithm_config: Union[
    "DiscreteRandomConfig",
    "SimpleLearningProgressConfig", 
    "BidirectionalLearningProgressConfig",
    "LearningProgressConfig"  # Current system
] = Field(default_factory=lambda: DiscreteRandomConfig())
```

## Success Criteria

### Phase 1 Success (Simple Learning Progress)
- [ ] SimpleLearningProgressAlgorithm passes all existing learning progress tests
- [ ] Performance is comparable to pre-dehydration minimal implementation
- [ ] Navigation recipe `train_simple_lp` runs successfully
- [ ] Learning progress stats are logged correctly

### Phase 2 Success (Bidirectional Learning Progress)  
- [ ] BidirectionalLearningProgressAlgorithm achieves O(1) task type indexing
- [ ] Performance matches or exceeds pre-dehydration sophisticated implementation
- [ ] All original algorithm features working (progress smoothing, sigmoid normalization)
- [ ] Navigation recipe `train_bidirectional_lp` shows improved convergence over baseline

### Overall Success
- [ ] All four navigation recipe variants run successfully
- [ ] Performance comparison shows bidirectional > simple > current > baseline
- [ ] Memory usage remains bounded for all variants
- [ ] Learning progress functionality restored to pre-dehydration capability

## Implementation Order

1. **Simple Learning Progress** (1-2 days)
   - Port minimal learning progress logic
   - Create SimpleLearningProgressConfig/Algorithm
   - Add train_simple_lp navigation recipe
   - Test and validate performance

2. **BidirectionalLearningProgress Port** (2-3 days)
   - Create TaskTypeMapper for O(1) indexing
   - Port BidirectionalLearningProgress algorithm
   - Integrate with task type enumeration
   - Add train_bidirectional_lp navigation recipe

3. **Testing and Validation** (1 day)
   - Add all recipe variants to navigation.py
   - Create comprehensive test suite
   - Performance comparison across all variants
   - Documentation updates

## Risk Mitigation

### Performance Risks
- **Task type enumeration complexity**: BucketedTaskGenerator already implements this efficiently
- **Memory usage**: All variants use bounded memory with proper cleanup
- **Integration overhead**: Minimal changes to existing Curriculum class

### Compatibility Risks
- **Configuration changes**: All changes are additive, existing configs remain valid
- **API stability**: No breaking changes to public Curriculum API
- **Rollback plan**: All variants can be disabled by using DiscreteRandomConfig

## Testing Strategy

### Unit Tests
- Test all learning progress algorithms in isolation
- Verify EMA calculations match pre-dehydration implementation
- Test task type mapping and indexing

### Integration Tests
- Test all navigation recipe variants
- Performance comparison between variants
- Memory usage validation over long runs

### Regression Tests
- Ensure existing functionality unchanged
- Current learning progress algorithm still works
- No performance degradation in baseline cases

This plan provides a clear path to restore pre-dehydration learning progress functionality while maintaining compatibility with the current system and providing multiple variants for comparison and optimization.