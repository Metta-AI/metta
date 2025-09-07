# Adaptive Experiment Infrastructure: Minimal Refactor Plan

## Executive Summary

After careful analysis, the current sweep infrastructure is already remarkably flexible. The key insight is that **we don't need to add features, but rather generalize the mental model** from "sweep" (hyperparameter optimization) to "adaptive experiment" (any controlled sequence of runs with feedback loops). This requires minimal but strategic changes.

## Core Abstraction: From Sweep to Experiment

### Current State Analysis

The existing architecture already has:
1. **Flexible Scheduler Protocol** - Can implement any sequencing logic
2. **Generic Optimizer Protocol** - Not limited to hyperparameters; can suggest any configuration
3. **Polymorphic JobDefinition** - Already handles both training and evaluation jobs
4. **Adaptive Control Loop** - Controller already monitors and adapts based on results

What's limiting is the **nomenclature and mental model**, not the architecture.

## Minimal Changes for Maximum Generalization

### 1. Model Changes (Semantic, not Structural)

```python
# metta/experiment/models.py (renamed from metta/sweep/models.py)

class ExperimentType(StrEnum):
    """Type of adaptive experiment being run"""
    HYPERPARAMETER_OPTIMIZATION = auto()  # Traditional sweep
    VALIDATION = auto()                   # Reproducibility validation
    ABLATION = auto()                     # Component ablation study
    CURRICULUM = auto()                   # Curriculum learning sequence
    POPULATION = auto()                   # Population-based training
    CUSTOM = auto()                       # User-defined experiment logic

@dataclass
class ExperimentMetadata:
    """Metadata for an adaptive experiment (formerly SweepMetadata)"""
    experiment_id: str
    experiment_type: ExperimentType
    # All existing SweepMetadata fields remain unchanged
    
@dataclass
class ExperimentConfig:
    """Configuration that defines the experiment's behavior"""
    type: ExperimentType
    objective: dict  # Generalized from just "metric" and "goal"
    parameters: dict  # Can be hyperparameters, seeds, architectures, etc.
    constraints: dict = field(default_factory=dict)  # Budget, time, resources
    
# JobDefinition remains EXACTLY the same - it's already generic
# RunInfo remains EXACTLY the same - it's already generic
# Observation becomes more general but keeps same structure
```

### 2. Protocol Changes (Clarification, not Modification)

```python
# metta/experiment/protocols.py

class ExperimentScheduler(Protocol):
    """
    Schedules jobs in an adaptive experiment.
    This is just a renamed Scheduler with clearer docstring.
    
    Examples:
    - HyperparameterScheduler: Suggests new hyperparameter configs
    - ValidationScheduler: Runs same config with different seeds
    - AblationScheduler: Systematically removes components
    - CurriculumScheduler: Sequences tasks by difficulty
    """
    # Method signature UNCHANGED
    def schedule(...) -> list[JobDefinition]: ...

class ConfigurationGenerator(Protocol):
    """
    Generates configurations for experiment runs.
    This is just a renamed Optimizer with broader scope.
    
    Examples:
    - ProteinOptimizer: Suggests hyperparameters
    - SeedGenerator: Provides deterministic seed sequences
    - ArchitectureGenerator: Proposes model architectures
    """
    # Method signature UNCHANGED but semantics broader
    def suggest(...) -> list[dict[str, Any]]: ...

# Dispatcher and Store remain EXACTLY the same
```

### 3. Controller Changes (Minimal)

```python
# metta/experiment/controller.py

class AdaptiveExperimentController:
    """Controller for adaptive experiments (formerly SweepController)"""
    
    def __init__(
        self,
        experiment_id: str,  # Renamed from sweep_id
        experiment_type: ExperimentType,  # NEW: Makes intent explicit
        scheduler: ExperimentScheduler,  # Renamed from Scheduler
        config_generator: ConfigurationGenerator,  # Renamed from optimizer
        # Everything else UNCHANGED
        ...
    ):
        # Implementation remains 99% the same
        # Just add experiment_type to metadata for monitoring
```

### 4. Tool Changes (Surface-level)

```python
# metta/tools/experiment.py (renamed from sweep.py)

class AdaptiveExperimentTool(Tool):
    """Tool for running adaptive experiments"""
    
    # Key change: explicit experiment type selection
    experiment_type: ExperimentType = ExperimentType.HYPERPARAMETER_OPTIMIZATION
    
    # Everything else can stay the same initially
    # sweep_name -> experiment_name
    # protein_config -> experiment_config (but keeps same structure)
    
    def invoke(self, args, overrides):
        # Select appropriate scheduler based on experiment_type
        scheduler = self._create_scheduler()
        
        # Select appropriate config generator
        config_gen = self._create_config_generator()
        
        # Rest remains the same

# Backward compatibility
SweepTool = AdaptiveExperimentTool  # Alias for migration
```

### 5. Monitor/Observatory Changes

The beauty is that **the Observatory already visualizes runs generically**. Changes needed:

```typescript
// observatory/src/components/ExperimentView.tsx

interface ExperimentMetadata {
  experimentId: string;
  experimentType: 'hyperparameter' | 'validation' | 'ablation' | 'curriculum' | 'population';
  // ... existing fields
}

// Add experiment-type-specific visualizations
const ExperimentDashboard = ({ metadata }) => {
  switch (metadata.experimentType) {
    case 'hyperparameter':
      return <HyperparameterChart />;  // Existing sweep viz
    case 'validation':
      return <ValidationMatrix />;      // Seed x Config matrix
    case 'ablation':
      return <AblationTree />;         // Component hierarchy
    default:
      return <GenericRunTable />;       // Fallback to current view
  }
};
```

## Implementation Strategy

### Phase 1: Core Renaming (No Logic Changes)
1. Copy `metta/sweep/` to `metta/experiment/`
2. Update imports gradually
3. Add deprecation warnings to old paths
4. **Zero functional changes** - pure refactoring

### Phase 2: Add Experiment Types
1. Implement `ExperimentType` enum
2. Add type-specific schedulers (ValidationScheduler, AblationScheduler)
3. Keep ProteinOptimizer as-is for HYPERPARAMETER_OPTIMIZATION type
4. Add simple ConfigurationGenerators for other types

### Phase 3: Specialize Visualizations
1. Add experiment type to metadata
2. Create type-specific dashboard components
3. Keep existing visualizations as default

## Key Insight: It's Already Adaptive

The current infrastructure is **already an adaptive experiment system**. The changes needed are:

1. **Naming**: Reflect the true generality of the system
2. **Mental Model**: Help users understand they can implement ANY experiment type
3. **Examples**: Provide concrete scheduler implementations for common patterns

## What We're NOT Changing

1. **Core Protocols**: The Store, Dispatcher protocols are perfect as-is
2. **Job Execution**: JobDefinition and dispatch mechanisms work for any experiment
3. **Monitoring Loop**: The adaptive control loop is already general
4. **Data Flow**: RunInfo and Observation are already flexible enough

## Concrete Example: Validation Experiment

With these minimal changes, validation becomes natural:

```python
class ValidationScheduler:
    """Runs same configurations with different seeds"""
    
    def schedule(self, metadata, all_runs, dispatched_trainings, dispatched_evals):
        jobs = []
        for config in self.target_configs:
            for seed in self.seed_sequence:
                if not self._already_run(config, seed, all_runs):
                    jobs.append(JobDefinition(
                        run_id=f"val_{config_id}_seed_{seed}",
                        cmd=self.train_recipe,
                        overrides={**config, 'system.seed': seed}
                    ))
        return jobs

class SeedGenerator:
    """Generates deterministic seed sequences"""
    
    def suggest(self, observations, n_suggestions=1):
        # Return next seeds in sequence
        return [{'seed': s} for s in self.seed_sequence[:n_suggestions]]
```

## Migration Path

### Week 1: Preparation
- Create parallel `metta/experiment/` structure
- Add compatibility aliases
- Update documentation to use new terminology

### Week 2: Gradual Migration
- New experiments use `AdaptiveExperimentTool`
- Existing sweeps continue working via aliases
- Monitor for any issues

### Week 3: Complete Transition
- Update all internal references
- Deprecate old names with warnings
- Release new Observatory UI

## Success Metrics

1. **Zero Breaking Changes**: All existing sweeps run unchanged
2. **Improved Clarity**: New users understand the system is general-purpose
3. **Same Performance**: No overhead from generalization
4. **Broader Adoption**: Teams use it for validation, ablation, curriculum experiments

## Conclusion

The current sweep infrastructure is **already a general adaptive experiment system**. We just need to:
1. **Rename** components to reflect their true generality
2. **Add explicit experiment types** to guide users
3. **Provide example schedulers** for common patterns

This is a refactoring that **reveals existing capability** rather than adding new complexity. The system's flexibility is already there - we're just making it obvious.