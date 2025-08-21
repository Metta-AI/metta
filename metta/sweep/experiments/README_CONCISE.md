# Next-Generation Experiment Orchestration Framework

## Quick Context (for Claude/AI)

- Building on PR#2088's execution layer with our orchestration layer
- Lives under `metta/sweep/` namespace
- Key insight: Experiments can contain experiments recursively
- `through` declarations = typed contracts + hook points
- Not overkill - essential complexity every ML team needs
- Status: Design complete, awaiting PR#2088 merge

## What Is This?

A modular framework that separates **what to try** (optimizers), **how to run it** (schedulers), and **how it flows** (pipelines). Instead of monolithic sweep scripts, you compose reusable pieces.

**Current System**: One big sweep program that does everything
**New System**: Lego blocks you can mix and match

## Why Should You Care?

Every ML team eventually needs:
- **Debugging** - "Why did my 6-hour run fail?"
- **Resumption** - "Can I continue from where it crashed?"
- **Comparison** - "Is Bayesian optimization better than random search for my problem?"
- **Scaling** - "How do I run 1000 experiments efficiently?"

This framework makes these problems go away by design, not as an afterthought.

## Core Concepts

### 1. Everything is an Experiment
- Training a model? That's an experiment
- Running a sweep? That's an experiment containing training experiments  
- Running curriculum learning? That's an experiment containing sweep experiments
- Optimizing your optimizer? That's an experiment containing experiments containing experiments

### 2. Three Layers That Work Together

**Optimizer**: Suggests what to try next
- Bayesian optimization, random search, evolutionary algorithms
- Plug in Ax, Optuna, Nevergrad, or your custom optimizer

**Scheduler**: Manages how things run
- Run in parallel batches? Use `BatchScheduler`
- Stop bad trials early? Use `ASHAScheduler`
- Evolve a population? Use `PBTScheduler`

**Pipeline**: Defines the flow with contracts
```python
Pipeline()
    .stage("suggest", get_suggestions)
    .through(trials=list[Trial])  # <-- Contract: next stage gets trials
    .stage("train", run_training)
    .through(results=dict)         # <-- Contract: next stage gets results
    .stage("update", update_model)
```

### 3. Contracts Make Errors Impossible

The `through` declarations:
- **Type-check** data between stages
- **Document** what flows where
- **Catch errors** before you run
- **Enable hooks** for monitoring, retries, checkpoints

### 4. Everything is Optimizable

Because experiments can contain experiments, you can optimize:
- Hyperparameters (classic)
- The optimization algorithm itself
- The curriculum sequence
- The resource allocation strategy
- The entire pipeline structure

## Key Benefits

1. **Mix and Match**: Swap optimizers/schedulers with one line
2. **Catch Errors Early**: Contracts validate before running
3. **Debug Easily**: Hooks show exactly what's happening
4. **Resume from Crashes**: Automatic checkpointing
5. **Scale Naturally**: Same code works locally or distributed
6. **Optimize Anything**: Even optimize the optimizers

## Simple Example

```python
# Old way: Monolithic sweep script with everything hardcoded

# New way: Compose what you need
experiment = SweepExperiment(
    optimizer=AxOptimizer(search_space),      # What to try
    scheduler=ASHAScheduler(max_epochs=100),  # How to run it
)
results = experiment.run()
```

## Advanced: Meta-Learning

Since experiments can contain experiments:

```python
# Level 1: Optimize hyperparameters
sweep = SweepExperiment(optimizer=BayesianOptimizer())

# Level 2: Optimize the optimizer
meta_sweep = MetaExperiment(
    base=SweepExperiment,
    optimizer=PBTScheduler(),  # Use PBT to find best Bayesian config
)

# Level 3: Keep going as deep as you want...
```

## Migration Path

Current code:
```bash
./tools/sweep_execute.py sweep_name=my_sweep
```

New code:
```python
SweepExperiment(
    optimizer=ProteinOptimizer(...),  # Same optimizer, wrapped
    scheduler=BatchScheduler(...),     # Explicit scheduling
).run()
```

---

## Code Examples

### Basic Sweep with Different Strategies

#### Bayesian Optimization
```python
from metta.sweep.experiments import SweepExperiment
from metta.sweep.optimizers import AxOptimizer
from metta.sweep.schedulers import SynchronousBatchScheduler

experiment = SweepExperiment(
    name="bayesian_sweep",
    optimizer=AxOptimizer(
        search_space={
            "learning_rate": LogUniform(1e-4, 1e-1),
            "batch_size": Choice([32, 64, 128, 256]),
            "hidden_size": Integer(64, 512),
        },
        acquisition="EI",  # Expected Improvement
    ),
    scheduler=SynchronousBatchScheduler(batch_size=10),
)
results = experiment.run(budget=100)
```

#### ASHA (Early Stopping)
```python
experiment = SweepExperiment(
    name="asha_sweep",
    optimizer=OptunaOptimizer(search_space, sampler="TPE"),
    scheduler=ASHAScheduler(
        max_epochs=100,
        reduction_factor=3,  # Keep top 1/3 at each rung
        grace_period=10,     # Min epochs before pruning
    ),
)
results = experiment.run()
```

#### Population-Based Training
```python
experiment = SweepExperiment(
    name="pbt_sweep",
    optimizer=RandomOptimizer(search_space),
    scheduler=PBTScheduler(
        population_size=20,
        num_generations=50,
        exploit_method="truncation",
        explore_method="perturb",
    ),
)
results = experiment.run()
```

### Pipeline with Contracts and Hooks

```python
class MyExperiment(Experiment):
    def pipeline(self):
        return (
            Pipeline()
            # Stage 1: Generate suggestions
            .stage("suggest", self.generate_suggestions)
            .through(
                SuggestionOutput,  # Type contract
                # Hooks for monitoring and control
                before=lambda d: logger.info(f"Generating {len(d.trials)} trials"),
                validate=lambda d: assert len(d.trials) > 0,
                checkpoint=True,
                timeout=300,
            )
            
            # Stage 2: Train models
            .stage("train", self.train_models)
            .through(
                TrainingOutput,
                retry=3,  # Retry failed training
                on_error=lambda e: self.handle_failure(e),
            )
            
            # Stage 3: Evaluate results
            .stage("evaluate", self.evaluate_results)
            .through(
                EvalOutput,
                cache=True,  # Cache expensive evaluations
            )
        )
```

### Recursive Experiments (Curriculum Learning)

```python
class CurriculumExperiment(Experiment):
    """Curriculum that runs a sweep for each task"""
    
    def pipeline(self):
        return (
            Pipeline()
            .stage("init", self.setup_curriculum)
            .through(tasks=list[Task])
            
            .loop_while(lambda ctx: ctx.task_idx < len(ctx.tasks))
                .stage("select_task", self.get_next_task)
                .through(current_task=Task)
                
                # This stage runs a complete sweep experiment!
                .stage("optimize_task", self.run_task_sweep)
                .through(task_results=dict)
                
                .stage("check_mastery", self.evaluate_progress)
                .through(should_advance=bool)
            .end_loop()
        )
    
    async def run_task_sweep(self, input):
        """Each task gets its own optimization"""
        sweep = SweepExperiment(
            optimizer=self.create_optimizer_for_task(input.current_task),
            scheduler=ASHAScheduler(),
        )
        return await sweep.run()
```

### Meta-Optimization (Optimizing the Optimizer)

```python
class OptimizerOptimizer(Experiment):
    """Find the best optimizer configuration"""
    
    def pipeline(self):
        return (
            Pipeline()
            # Suggest optimizer configuration
            .stage("suggest_config", self.suggest_optimizer_params)
            .through(
                acquisition_fn=str,
                exploration_rate=float,
                batch_size=int,
            )
            
            # Run a sweep with that configuration
            .stage("test_optimizer", self.run_sweep_with_config)
            .through(performance=float)
            
            # Update our meta-optimizer
            .stage("update", self.update_meta_model)
        )
    
    async def run_sweep_with_config(self, config):
        # Create optimizer with suggested params
        optimizer = create_optimizer(
            acquisition=config.acquisition_fn,
            exploration=config.exploration_rate,
        )
        
        # Run full sweep to test this optimizer
        sweep = SweepExperiment(optimizer=optimizer)
        result = await sweep.run()
        return result.best_score
```

### Advanced Hook Patterns

```python
# Reusable hook sets
debug_hooks = {
    "before": lambda d: print(f"Input: {d}"),
    "after": lambda d: print(f"Output: {d}"),
    "validate": lambda d: validate_debug(d),
}

production_hooks = {
    "metrics": lambda d: wandb.log(d),
    "checkpoint": True,
    "retry": 3,
    "on_error": lambda e: alert_team(e),
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

# Global hooks for all stages
pipeline = pipeline.with_global_hooks(
    before=lambda stage, data: logger.info(f"[{stage}] Starting"),
    after=lambda stage, data: metrics.record(f"{stage}.complete"),
    profile=True,  # Profile everything
)
```

### Composing Complex Strategies

```python
class HybridExperiment(Experiment):
    """Combine multiple optimization strategies"""
    
    def pipeline(self):
        return (
            Pipeline()
            # Phase 1: Broad exploration with random search
            .stage("explore", self.random_search)
            .through(
                exploration_results=list[Trial],
                n_trials=50,
                max_time_per_trial=600,  # Quick eval
            )
            
            # Phase 2: Model the space with Bayesian optimization
            .stage("model", self.bayesian_optimization)
            .through(
                modeled_results=list[Trial],
                n_trials=100,
                init_from=exploration_results,  # Warm start
            )
            
            # Phase 3: Refine with population evolution
            .stage("refine", self.population_evolution)
            .through(
                final_results=list[Trial],
                population_size=20,
                generations=30,
            )
            
            # Phase 4: Validate best candidates
            .stage("validate", self.intensive_evaluation)
            .through(
                validation_scores=dict,
                n_seeds=10,
                full_evaluation=True,
            )
        )
```

### Custom Optimizers and Schedulers

```python
# Custom optimizer
class MyOptimizer(Optimizer):
    def suggest(self, n: int) -> list[Trial]:
        # Your suggestion logic
        return trials
    
    def observe(self, trial: Trial, result: float):
        # Update your model
        pass

# Custom scheduler
class MyScheduler(Scheduler):
    def should_stop(self, trial: Trial) -> bool:
        # Your stopping logic
        return False
    
    def allocate_resources(self, trial: Trial) -> Resources:
        # Your resource allocation
        return Resources(gpus=1, time_limit=3600)

# Use them like any other
experiment = SweepExperiment(
    optimizer=MyOptimizer(),
    scheduler=MyScheduler(),
)
```