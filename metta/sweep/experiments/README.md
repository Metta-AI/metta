# Next-Generation Sweep Orchestration Framework

## Context for Claude

This document describes a next-generation experiment orchestration framework being developed for Metta. Key context:

- **Current State**: We have an existing sweep system using Protein optimizer with monolithic design
- **Integration with PR#2088**: Jack Heart's experiments system provides the execution layer; our framework adds the orchestration/optimization layer on top
- **Philosophy**: Not just for sweeps - this is a general recursive experiment composition framework where sweeps, curricula, and meta-learning are different ways of composing experiments
- **Key Innovation**: The `through` declarations create typed contracts AND programmable hook points between pipeline stages
- **Design Decisions**: 
  - Explicit contracts are intentional - catching errors at definition time is worth the verbosity
  - Recursive composition is essential - experiments can contain experiments arbitrarily
  - Hooks at stage boundaries provide observability without modifying business logic
  - This is NOT overkill - it's essential complexity that every ML team eventually needs
- **Implementation Status**: Design phase complete, waiting for PR#2088 to merge before implementation
- **Namespace**: Everything lives under `metta/sweep/` to maintain clear organization

## Why This Isn't Overkill

Every production ML team eventually builds these features, but usually as painful retrofits:

1. **The Debugging Tax**: Ad-hoc print statements vs systematic observability through hooks
2. **The Silent Failure Problem**: ML fails subtly - contracts catch these early
3. **The Reproducibility Crisis**: Without checkpointing and explicit data flow, experiments become unreproducible
4. **The Monitoring Afterthought**: Teams add metrics after failures - hooks make it first-class from day one

The framework provides essential complexity, not accidental complexity. Each feature addresses real pain points that emerge in any serious ML research effort. The fact that simple cases remain simple (`Pipeline().stage("train", train).stage("eval", eval)`) while complex cases become possible shows this is the right level of abstraction.

## Overview

The new sweep orchestration framework separates concerns between **optimization algorithms**, **scheduling strategies**, and **experiment flow**. This modular architecture allows researchers to mix and match components freely, creating sophisticated hyperparameter optimization strategies with minimal code changes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Layer                         │
│  (Defines flow through composable stages)                   │
├─────────────────────────────────────────────────────────────┤
│                   Optimizer Layer                           │
│  (Suggests parameters: Bayesian, Random, Evolutionary)      │
├─────────────────────────────────────────────────────────────┤
│                   Scheduler Layer                           │
│  (Manages execution: ASHA, PBT, Hyperband, Batch)          │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **Experiment**: Orchestrates the overall flow through staged pipelines
- **Optimizer**: Suggests hyperparameters based on observed results
- **Scheduler**: Manages trial execution, resources, and pruning decisions
- **Executor**: Handles the actual training job execution (local, SLURM, cloud)

## Examples

### Classic Bayesian Optimization

Traditional sequential Bayesian optimization with batch parallelism:

```python
from metta.sweep.experiments import SweepExperiment
from metta.sweep.optimizers import AxOptimizer
from metta.sweep.schedulers import SynchronousBatchScheduler

experiment = SweepExperiment(
    name="bayesian_optimization",
    optimizer=AxOptimizer(
        search_space={
            "trainer.learning_rate": LogUniform(1e-4, 1e-1),
            "trainer.batch_size": Choice([32, 64, 128, 256]),
            "agent.hidden_size": Integer(64, 512),
        },
        acquisition="EI",  # Expected Improvement
    ),
    scheduler=SynchronousBatchScheduler(
        batch_size=10,     # Run 10 trials in parallel
        max_parallel=10,   # Resource limit
    ),
)

results = experiment.run(budget=100)  # Run 100 trials total
```

### ASHA (Asynchronous Successive Halving)

Early stopping with promotion through performance rungs:

```python
from metta.sweep.optimizers import OptunaOptimizer
from metta.sweep.schedulers import ASHAScheduler

experiment = SweepExperiment(
    name="asha_tpe",
    optimizer=OptunaOptimizer(
        search_space=search_space,
        sampler="TPE",  # Tree-structured Parzen Estimator
    ),
    scheduler=ASHAScheduler(
        max_epochs=100,        # Maximum training epochs
        reduction_factor=3,    # Keep top 1/3 at each rung
        grace_period=10,       # Min epochs before pruning
    ),
)

# ASHA automatically promotes/prunes trials based on intermediate performance
results = experiment.run(budget=500)
```

### Population-Based Training (PBT)

Evolving population of models with online hyperparameter adaptation:

```python
from metta.sweep.schedulers import PBTScheduler

experiment = SweepExperiment(
    name="pbt_evolution",
    optimizer=RandomOptimizer(search_space),  # PBT handles exploitation
    scheduler=PBTScheduler(
        population_size=20,
        perturbation_factor=0.2,
        num_generations=50,
        exploit_method="truncation",  # Top 20% reproduce
        explore_method="perturb",      # Random perturbation
    ),
)

# PBT evolves hyperparameters DURING training
results = experiment.run()
```

### Hyperband

Principled early stopping with multiple brackets:

```python
from metta.sweep.schedulers import HyperbandScheduler

experiment = SweepExperiment(
    name="hyperband_optimization",
    optimizer=NevergradOptimizer(
        search_space=search_space,
        algorithm="PSO",  # Particle Swarm Optimization
    ),
    scheduler=HyperbandScheduler(
        max_resource=81,  # e.g., epochs
        eta=3,            # Downsampling rate
    ),
)

results = experiment.run()
```

### Hybrid Multi-Phase Strategy

Combining different strategies for different phases:

```python
from metta.sweep.experiments.base import Experiment, Pipeline

class HybridSweepExperiment(Experiment):
    """
    Phase 1: Random exploration to map the space
    Phase 2: Bayesian optimization to model promising regions  
    Phase 3: Population evolution for final refinement
    """
    
    def pipeline(self):
        return (
            Pipeline()
            # Broad exploration with random search
            .stage("explore", self.random_search(
                n_trials=50,
                resources_per_trial=0.5,  # Quick evaluations
            ))
            
            # Focused search with Gaussian Process model
            .stage("model", self.bayesian_optimize(
                n_trials=100,
                acquisition="UCB",
                beta=2.0,  # Exploration/exploitation tradeoff
            ))
            
            # Population refinement in best region
            .stage("evolve", self.population_training(
                population_size=20,
                generations=30,
            ))
            
            # Final validation of best candidates
            .stage("validate", self.final_evaluation(
                n_seeds=10,  # Multiple seeds for robustness
            ))
        )
```

## Benefits Over Current System

### 1. Flexibility

**Current System**: Monolithic sweep with fixed Protein optimizer and batch scheduling

**New System**: Mix any optimizer with any scheduler:
```python
# Easy to experiment with different combinations
for optimizer in [AxOptimizer, OptunaOptimizer, NevergradOptimizer]:
    for scheduler in [ASHAScheduler, PBTScheduler, HyperbandScheduler]:
        experiment = SweepExperiment(
            optimizer=optimizer(search_space),
            scheduler=scheduler(max_resource=100),
        )
        results.append(experiment.run())
```

### 2. Clarity

**Current System**: Complex lifecycle functions with implicit data flow

**New System**: Explicit stages with clear data contracts:
```python
@dataclass
class SweepContext:
    """Type-safe context flowing through pipeline"""
    trials: list[Trial]
    observations: list[Observation]
    current_phase: str
    
async def suggest_stage(ctx: SweepContext) -> SweepContext:
    """Each stage has explicit input/output"""
    ctx.trials = self.optimizer.suggest(n=10)
    return ctx
```

### 3. Extensibility

**Current System**: Adding new strategies requires modifying core sweep code

**New System**: Implement simple interfaces:
```python
class MyOptimizer(Optimizer):
    def suggest(self, n: int) -> list[Trial]:
        # Your suggestion logic
        
    def observe(self, trial: Trial, result: float):
        # Update your model

class MyScheduler(Scheduler):
    def should_stop(self, trial: Trial) -> bool:
        # Your pruning logic
        
    def allocate_resources(self, trial: Trial) -> Resources:
        # Your resource allocation
```

### 4. Composability

**Current System**: Fixed pipeline with phases

**New System**: Compose sophisticated strategies:
```python
pipeline = (
    Pipeline()
    .parallel([
        self.random_search(n=20),
        self.grid_search(params=critical_params),
        self.sobol_sequence(n=20),
    ])
    .stage("merge", self.combine_observations)
    .stage("model", self.fit_gaussian_process)
    .conditional(
        lambda ctx: ctx.best_score > threshold,
        Pipeline().stage("exploit", self.intensive_local_search),
        Pipeline().stage("explore", self.continue_global_search),
    )
)
```

### 5. Testing

**Current System**: End-to-end testing only

**New System**: Test each component independently:
```python
def test_optimizer():
    opt = AxOptimizer(search_space)
    trials = opt.suggest(10)
    assert len(trials) == 10
    
def test_scheduler():  
    sched = ASHAScheduler(max_t=100)
    assert sched.get_next_rung(10) == 30  # 10 * 3
    
def test_pipeline():
    ctx = SweepContext()
    ctx = await suggest_stage(ctx)
    assert len(ctx.trials) > 0
```

## Migration Guide

### Current Code
```python
# Old way using sweep_execute.py
./tools/sweep_execute.py sweep_name=my_sweep \
    sweep.protein.acquisition_fn=ucb \
    sweep.protein.ucb_beta=2.0
```

### New Code
```python
# New way with explicit components
from metta.sweep.experiments import SweepExperiment
from metta.sweep.optimizers import ProteinOptimizer
from metta.sweep.schedulers import BatchScheduler

experiment = SweepExperiment(
    name="my_sweep",
    optimizer=ProteinOptimizer(
        acquisition_fn="ucb",
        ucb_beta=2.0,
        # Same parameters, clearer structure
    ),
    scheduler=BatchScheduler(
        batch_size=10,
        # Scheduling is now explicit
    ),
)

experiment.run()
```

## Supported Strategies

### Optimizers
- **ProteinOptimizer**: Current Gaussian Process-based Bayesian optimization
- **AxOptimizer**: Facebook's Ax platform (BoTorch, SAASBO, etc.)
- **OptunaOptimizer**: Tree-structured Parzen Estimator, CMA-ES
- **NevergradOptimizer**: Evolutionary algorithms, PSO, differential evolution
- **RandomOptimizer**: Random search with various distributions
- **GridOptimizer**: Systematic grid search
- **SobolOptimizer**: Quasi-random Sobol sequences

### Schedulers
- **SynchronousBatchScheduler**: Wait for full batches to complete
- **AsynchronousScheduler**: Launch trials as resources become available
- **ASHAScheduler**: Asynchronous Successive Halving Algorithm
- **HyperbandScheduler**: Principled early stopping with multiple brackets
- **PBTScheduler**: Population-Based Training with online adaptation
- **PB2Scheduler**: Population-Based Bandits
- **MedianStoppingScheduler**: Stop trials below median performance

### Executors
- **LocalExecutor**: Run on local machine
- **SLURMExecutor**: Submit to SLURM cluster
- **CogwebExecutor**: Use Cogweb backend
- **DockerExecutor**: Run in containers
- **KubernetesExecutor**: Run on K8s cluster

## Advanced Patterns

### Type-Safe Pipeline Contracts

The framework supports explicit data contracts between pipeline stages, ensuring type safety and catching errors at definition time:

#### Basic Contract Declaration
```python
from typing import NamedTuple

# Define what flows between stages
class SuggestionOutput(NamedTuple):
    trials: list[Trial]
    batch_id: str

class TrainingOutput(NamedTuple):
    results: dict[str, float]
    checkpoints: dict[str, str]
    batch_id: str

# Pipeline with explicit contracts
pipeline = (
    Pipeline()
    .stage("suggest", self.generate_suggestions)
    .through(SuggestionOutput)  # Declares output type
    .stage("train", self.train_trials)  # Must accept SuggestionOutput
    .through(TrainingOutput)  # Declares output type
    .stage("evaluate", self.evaluate_results)  # Must accept TrainingOutput
)
```

#### Fluent Contract Syntax
```python
# Alternative syntax with inline type declarations
pipeline = (
    Pipeline()
    .stage("prepare", self.prepare_batch)
    .through(run_name=str, run_id=int, trials=list[Trial])
    .stage("train", self.train_batch)  # Verified to accept run_name, run_id, trials
    .through(results=dict[str, float], checkpoints=dict[str, str])
    .stage("evaluate", self.evaluate_results)  # Verified to accept results, checkpoints
)
```

#### Benefits of Explicit Contracts
- **Compile-time verification**: Type checkers verify the pipeline before runtime
- **Self-documenting**: Contracts serve as inline documentation
- **Refactoring-safe**: Changes to contracts immediately show what needs updating
- **IDE support**: Autocomplete knows exactly what data is available at each stage
- **Testable**: Stages can be tested in isolation with clear input/output specs

### Pipeline Hooks

The `through` declarations create natural interception points where hooks can be injected for monitoring, debugging, validation, and control flow:

#### Basic Hook Usage
```python
pipeline = (
    Pipeline()
    .stage("suggest", self.generate_suggestions)
    .through(
        SuggestionOutput,
        # Observation hooks
        before=lambda data: logger.info(f"Generating {len(data.trials)} trials"),
        after=lambda data: metrics.record("trials.generated", len(data.trials)),
        
        # Validation hooks
        validate=lambda data: assert len(data.trials) > 0,
        
        # Transformation hooks
        transform=lambda data: data._replace(trials=shuffle(data.trials)),
        
        # Persistence hooks
        checkpoint=True,  # Save output to disk
        cache=True,       # Cache for reuse
        
        # Control flow hooks
        timeout=300,      # Max 5 minutes for this stage
        retry=3,          # Retry up to 3 times on failure
    )
    .stage("train", self.train_trials)
    .through(
        TrainingOutput,
        on_error=lambda e: self.handle_training_failure(e),
        profile=True,  # Profile this stage for performance
    )
)
```

#### Types of Hooks

**Observation Hooks** (non-mutating):
```python
.through(
    OutputType,
    log=lambda data: logger.info(f"Stage output: {data}"),
    metrics=lambda data: wandb.log({"stage.output": data.metric}),
    debug=lambda data: breakpoint() if data.batch_id == "debug",
    monitor=lambda data: health_check.ping(data.batch_id),
)
```

**Validation Hooks**:
```python
.through(
    OutputType,
    validate_type=True,  # Automatic type validation
    validate=lambda data: validate_output(data),  # Custom validation
    schema=OutputSchema,  # Schema validation
    assert_fn=lambda data: assert data.metric > 0,  # Assertions
)
```

**Transformation Hooks**:
```python
.through(
    OutputType,
    transform=lambda data: normalize_output(data),
    filter=lambda data: filter_successful_trials(data),
    enrich=lambda data: add_metadata(data),
    serialize=lambda data: data.to_json(),
)
```

**Control Flow Hooks**:
```python
.through(
    OutputType,
    retry=RetryPolicy(max_attempts=3, backoff=exponential),
    on_error=lambda e, data: recover_from_error(e, data),
    skip_if=lambda data: len(data.trials) == 0,
    timeout=300,
)
```

**Persistence Hooks**:
```python
.through(
    OutputType,
    checkpoint=True,
    checkpoint_path=lambda data: f"checkpoints/{data.batch_id}",
    cache=True,
    cache_key=lambda data: hash(data.trials),
    archive=lambda data: s3.upload(data, f"runs/{data.batch_id}"),
)
```

#### Advanced Hook Patterns

**Composable Hook Sets**:
```python
# Define reusable hook configurations
debug_hooks = {
    "before": lambda d: logger.debug(f"Input: {d}"),
    "after": lambda d: logger.debug(f"Output: {d}"),
    "validate": lambda d: validate_debug_mode(d),
}

production_hooks = {
    "metrics": lambda d: record_metrics(d),
    "checkpoint": True,
    "retry": 3,
    "on_error": lambda e: alert_on_call(e),
}

# Apply conditionally
pipeline = (
    Pipeline()
    .stage("train", self.train)
    .through(
        TrainingOutput,
        **(debug_hooks if DEBUG else production_hooks)
    )
)
```

**Global Pipeline Hooks**:
```python
# Apply hooks to all stages
pipeline = pipeline.with_global_hooks(
    before=lambda stage, data: logger.info(f"[{stage}] Starting"),
    after=lambda stage, data: logger.info(f"[{stage}] Complete"),
    on_error=lambda stage, error: alert_team(f"Stage {stage} failed: {error}"),
    profile=True,  # Profile all stages
)
```

#### Benefits of Hook System
- **Observability**: Monitor data flow without modifying stage logic
- **Debuggability**: Inspect and breakpoint at stage boundaries
- **Resilience**: Automatic retries and error recovery
- **Performance**: Profiling and timeout management
- **Caching**: Avoid recomputation of expensive stages
- **Validation**: Ensure data contracts are met at runtime
- **Flexibility**: Add operational concerns without touching business logic

The `through` declarations become **programmable membranes** between stages, providing both type contracts and behavioral contracts that define how data flows, how it's validated, how errors are handled, and how the system observes itself.

### Recursive Experiment Composition

Experiments can nest arbitrarily, enabling sophisticated multi-level strategies:

```python
class CurriculumExperiment(Experiment):
    """Curriculum that orchestrates sweep experiments per task"""
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .stage("init", self.initialize_curriculum)
            .through(tasks=list[Task], current_stage=int)
            .loop_while(lambda ctx: ctx.current_stage < len(ctx.tasks))
                .stage("select_task", self.select_next_task)
                .through(current_task=Task)
                .stage("run_sweep", self.execute_task_sweep)  # Nested experiment!
                .through(task_results=dict)
                .stage("evaluate", self.assess_mastery)
                .through(should_advance=bool)
            .end_loop()
        )
    
    async def execute_task_sweep(self, input: TaskSelection) -> TaskResults:
        """Each curriculum stage runs a full sweep experiment"""
        sweep = SweepExperiment(
            optimizer=self.optimizer_for_task(input.current_task),
            scheduler=self.scheduler_for_task(input.current_task),
        )
        results = await sweep.run()  # Nested experiment execution
        return TaskResults(task_results=results)
```

### Warm-Starting
```python
# Use results from previous sweep
previous_observations = load_observations("previous_sweep")
optimizer = AxOptimizer(search_space)
for obs in previous_observations:
    optimizer.observe(obs.trial, obs.result)
    
experiment = SweepExperiment(optimizer=optimizer, ...)
```

### Multi-Objective Optimization
```python
optimizer = AxOptimizer(
    search_space=search_space,
    objectives=["accuracy", "latency"],
    objective_thresholds={"accuracy": 0.9, "latency": 100},
)
```

### Conditional Search Spaces
```python
search_space = {
    "model_type": Choice(["cnn", "transformer"]),
    "cnn.layers": Integer(3, 10),  # Only when model_type == "cnn"
    "transformer.heads": Integer(4, 16),  # Only when model_type == "transformer"
}
```

### Resource-Aware Scheduling
```python
scheduler = ResourceAwareScheduler(
    resource_attr="training_time",
    max_concurrent_trials=10,
    prioritize_by="performance_per_cost",
)
```

## Performance Considerations

- **Async Execution**: Non-blocking trial management for better resource utilization
- **Batching**: Amortize optimization overhead across multiple trials
- **Early Stopping**: Aggressively prune poor trials to focus resources
- **Caching**: Reuse expensive computations (GP fits, etc.)
- **Distributed**: Scale across multiple machines seamlessly

## Future Extensions

The framework is designed for future research directions:

- **Meta-Learning**: Learn optimizers from previous sweeps
- **Multi-Fidelity**: Combine cheap proxies with expensive evaluations
- **Transfer Learning**: Share knowledge across related tasks
- **Neural Architecture Search**: Specialized optimizers for NAS
- **AutoML**: Full pipeline optimization beyond hyperparameters
- **Federated**: Distributed optimization with privacy constraints

## Meta-Optimization: Optimizing the Optimizers

The recursive nature of this framework enables a profound capability: **everything becomes optimizable**. Not just hyperparameters, but the entire experimental process itself - optimizers, schedulers, curricula, even the pipeline structure.

### Levels of Optimization

The framework naturally supports arbitrary levels of meta-learning:

```python
# Level 0: Train a Model
TrainingExperiment()  # Fixed hyperparameters

# Level 1: Optimize Hyperparameters  
SweepExperiment(
    base_experiment=TrainingExperiment,
    optimizer=BayesianOptimizer(),
)

# Level 2: Optimize the Optimizer
MetaSweepExperiment(
    base_experiment=SweepExperiment,
    meta_optimizer=PBTScheduler(),  # Find best optimization strategy
)

# Level 3: Learn to Learn to Learn
MetaMetaExperiment(
    base_experiment=MetaSweepExperiment,
    meta_meta_optimizer=EvolutionaryStrategy(),  # Find best meta-optimization approach
)
```

### Example: Optimizing Curriculum Learning

Using PBT to discover the best task sequencing strategy:

```python
class CurriculumOptimizationExperiment(Experiment):
    """Use PBT to evolve curriculum schedulers"""
    
    def pipeline(self):
        return (
            Pipeline()
            .stage("init_population", self.create_curriculum_population)
            .through(population=list[CurriculumScheduler])
            .loop_while(lambda ctx: ctx.generation < 50)
                .stage("evaluate_curricula", self.run_curriculum_experiments)
                .through(curriculum_results=dict[str, float])
                .stage("evolve", self.evolve_curriculum_strategies)
            .end_loop()
        )
    
    async def run_curriculum_experiments(self, ctx):
        """Each curriculum scheduler is itself an experiment"""
        for scheduler in ctx.population:
            # Each scheduler determines a different task sequence
            curriculum_exp = CurriculumExperiment(
                scheduler=scheduler,
                base_experiment=TrainingExperiment,
            )
            result = await curriculum_exp.run()
            ctx.curriculum_results[scheduler.id] = result.final_performance
        return ctx
    
    def evolve_curriculum_strategies(self, ctx):
        """Evolve the scheduling strategies themselves"""
        # Top performing curricula reproduce with mutations
        # Task sequences, progression criteria, difficulty ramping all evolve
        return evolved_population
```

### Example: Learning to Optimize

Discovering the best optimization strategy for a problem class:

```python
class OptimizerOptimizationExperiment(Experiment):
    """Learn optimal optimizer configurations"""
    
    def pipeline(self):
        return (
            Pipeline()
            .stage("suggest_optimizer_config", self.meta_optimizer.suggest)
            .through(
                acquisition_fn=str,
                batch_size=int,
                exploration_rate=float,
                scheduler_type=str,
            )
            .stage("run_sweep", self.execute_sweep_with_config)
            .through(sweep_performance=float)
            .stage("update", self.meta_optimizer.observe)
        )
    
    async def execute_sweep_with_config(self, config):
        """Test the suggested optimizer configuration"""
        optimizer = create_optimizer(
            acquisition=config.acquisition_fn,
            exploration=config.exploration_rate,
        )
        
        # Run a full sweep with this optimizer
        sweep = SweepExperiment(
            optimizer=optimizer,
            scheduler=create_scheduler(config.scheduler_type),
        )
        result = await sweep.run()
        return SweepResult(sweep_performance=result.best_metric)
```

### What Becomes Optimizable

With recursive composition, every component of the experimental process can be optimized:

#### 1. **Task Scheduling Strategies**
```python
# Optimize how tasks are sequenced in curriculum learning
TaskSchedulerOptimization(
    strategies=[
        RandomScheduler(),
        DifficultyRampScheduler(),
        InterleaveScheduler(),
        AdaptiveScheduler(),
    ]
)
```

#### 2. **Pipeline Architecture**
```python
# Optimize the structure of the experiment itself
PipelineArchitectureSearch(
    search_space={
        "parallel_vs_sequential": Choice(["parallel", "sequential"]),
        "num_refinement_stages": Integer(1, 5),
        "checkpoint_frequency": Integer(1, 100),
    }
)
```

#### 3. **Resource Allocation Policies**
```python
# Learn optimal resource distribution across trials
ResourceSchedulerOptimization(
    policies=[
        EqualAllocation(),
        ProportionalToPromise(),
        AdaptiveBandit(),
    ]
)
```

#### 4. **Aggregation Strategies**
```python
# Discover best ways to combine results
AggregationOptimization(
    strategies=[
        MeanAggregator(),
        MedianAggregator(),
        BestOfN(n=3),
        WeightedByConfidence(),
        EnsembleVoting(),
    ]
)
```

### Implications for AGI Research

This recursive optimization capability enables:

1. **AutoML 2.0**: Not just automating model selection, but the entire experimental methodology
2. **Self-Improving Optimizers**: Optimizers that learn from their own performance
3. **Adaptive Curricula**: Task sequences that evolve based on learner characteristics
4. **Meta-Research Automation**: Systematically discovering what works across problem domains
5. **Learned Experimental Design**: AI systems that design their own experiments

The framework makes it possible to ask and answer questions like:
- "What's the best way to search for the best way to train this model?"
- "How should we schedule tasks to maximize hyperparameter optimization efficiency?"
- "What optimization algorithm works best for finding good optimization algorithms?"

This isn't just about finding good hyperparameters anymore - it's about **optimizing the entire scientific process** of machine learning research.

## Conclusion

The new sweep orchestration framework transforms hyperparameter optimization from a monolithic system into a **composable, extensible, and testable** platform. By separating optimization algorithms, scheduling strategies, and experiment flow, researchers can:

1. **Experiment faster** with different strategies
2. **Build sophisticated** multi-phase optimization pipelines
3. **Test components** in isolation for reliability
4. **Extend easily** with new optimizers and schedulers
5. **Share and reproduce** experiments with explicit configurations
6. **Optimize any aspect** of the experimental process itself

This positions Metta's sweep system at the forefront of hyperparameter optimization research, ready for the next generation of optimization challenges including meta-learning, AutoML, and ultimately AGI research where learning how to learn better becomes paramount.