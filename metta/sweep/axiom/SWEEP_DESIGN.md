# SWEEP_DESIGN.md

## tAXIOM Sweep Implementation Design

This document outlines the complete design for implementing hyperparameter sweeps using the tAXIOM DSL, incorporating all architectural decisions and improvements discussed.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Design Decisions](#core-design-decisions)
3. [Implementation Pattern](#implementation-pattern)
4. [Hook System](#hook-system)
5. [Context Management](#context-management)
6. [Complete Example](#complete-example)
7. [Migration Plan](#migration-plan)
8. [Implementation Roadmap](#implementation-roadmap)

## Architecture Overview

The sweep implementation follows a three-layer architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Configuration Layer                   │
│         (Frozen, Immutable, Dependency Injection)       │
├─────────────────────────────────────────────────────────┤
│                     Context Layer                        │
│          (Mutable Flow State, Resource Access)          │
├─────────────────────────────────────────────────────────┤
│                     Pipeline Layer                       │
│            (Stages, I/O, Hooks, Control Flow)           │
└─────────────────────────────────────────────────────────┘
```

## Core Design Decisions

### 1. Configuration vs Context Separation

**Configuration (Immutable)**
- All service objects (optimizer, logger, database)
- Base configurations (training, evaluation)
- Experiment parameters (max_trials, budget)

**Context (Mutable)**
- Flow control state (trial_id, remaining_budget)
- Accumulated history
- References immutable config

### 2. Stage vs I/O vs Hook Classification

```
Operation Decision Tree:
├── Removable without breaking? → Hook
├── Predictable output? 
│   ├── Yes → Stage
│   └── No → I/O
```

### 3. Method-Based Operations

Instead of lambdas everywhere, operations are defined as methods on the experiment class:
- Cleaner syntax
- Better testability
- IDE support
- Reusability

### 4. Decorator-Based Hooks

Hooks can be attached via decorators for cleaner separation of concerns.

## Implementation Pattern

### Base Classes

```python
from dataclasses import dataclass, field
from typing import Protocol, Any
from pydantic import BaseModel

# Protocols for service interfaces
class OptimizerProtocol(Protocol):
    def suggest(self) -> Suggestion: ...
    def observe(self, suggestion: Suggestion, score: float, cost: float): ...
    def load_observations(self, observations: list[Observation]): ...

class DatabaseClient(Protocol):
    def register_sweep(self, name: str) -> SweepRegistration: ...
    def fetch_observations(self, sweep_name: str) -> list[Observation]: ...
    def save_trial(self, sweep_name: str, trial_id: int, results: Any): ...
    def mark_complete(self, sweep_name: str): ...

class MetricLogger(Protocol):
    def log(self, metrics: dict[str, Any]): ...
    def log_metric(self, name: str, value: float): ...
    def log_timing(self, stage: str, elapsed: float): ...

class Monitor(Protocol):
    def log(self, message: str): ...
    def alert(self, message: str): ...
```

### Configuration Structure

```python
@dataclass(frozen=True)
class SweepExperimentConfig(BaseModel):
    """Immutable configuration containing all dependencies and settings."""
    
    # Base configurations (data)
    train_base_cfg: TrainConfig
    eval_base_cfg: EvalConfig
    
    # Service objects (interfaces)
    optimizer: OptimizerProtocol
    db_client: DatabaseClient
    metric_logger: MetricLogger
    monitor: Monitor
    
    # Experiment parameters
    max_trials: int = 50
    budget: float = 1e9
    convergence_threshold: float = 0.95
    
    # Metadata
    sweep_name: str
    sweep_worker_id: str
    random_seed: int = 42
```

### Context Structure

```python
@dataclass
class SweepExperimentCtx:
    """Mutable context with flow state."""
    config: SweepExperimentConfig
    
    # Mutable flow state
    current_trial_idx: int = 0
    remaining_budget: float = field(init=False)
    history: list[Observation] = field(default_factory=list)
    best_score: float = -float('inf')
    converged: bool = False
    
    def __post_init__(self):
        self.remaining_budget = self.config.budget
    
    def update(self, **kwargs):
        """Update mutable state."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Cannot update non-existent field: {key}")
        return self
```

### Pipeline Implementation

```python
class Pipeline:
    """Enhanced pipeline with new features."""
    
    def __init__(self, ctx: ExperimentCtx = None):
        self.ctx = ctx
        self.operations = []
    
    def stage(self, name: str, func: Callable) -> 'Pipeline':
        """Add a predictable, idempotent computation."""
        self.operations.append(('stage', name, func))
        return self
    
    def io(self, name: str, func: Callable) -> 'Pipeline':
        """Add an unpredictable I/O operation."""
        self.operations.append(('io', name, func))
        return self
    
    def through(self, output_type: type, 
                input_type: type = None, 
                hooks: list[Callable] = None) -> 'Pipeline':
        """Define type contracts and hooks at membrane."""
        last_op = self.operations[-1]
        last_op = (*last_op, {'output_type': output_type, 
                              'input_type': input_type, 
                              'hooks': hooks or []})
        self.operations[-1] = last_op
        return self
    
    def T(self, output_type: type) -> 'Pipeline':
        """Shorthand for through(output_type)."""
        return self.through(output_type)
    
    def hook(self, func: Callable) -> 'Pipeline':
        """Add a removable observer."""
        self.operations.append(('hook', None, func))
        return self
    
    def do_while(self, condition: Callable, inner: 'Pipeline') -> 'Pipeline':
        """Loop with condition check and automatic context updates."""
        self.operations.append(('do_while', condition, inner))
        return self
    
    def do_until(self, condition: Callable, inner: 'Pipeline') -> 'Pipeline':
        """Loop until condition met."""
        self.operations.append(('do_until', condition, inner))
        return self
    
    def run(self) -> Any:
        """Execute the pipeline."""
        # Implementation details omitted for brevity
        pass
```

## Hook System

### Decorator-Based Hooks

```python
from functools import wraps
import time
from typing import Callable, Any

def log_before(message: str | Callable):
    """Log before execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            msg = message(self.ctx) if callable(message) else message
            self.ctx.config.monitor.log(msg)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def log_after(formatter: Callable[[Any, Any], str]):
    """Log after execution with result."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.ctx.config.monitor.log(formatter(result, self.ctx))
            return result
        return wrapper
    return decorator

def timed(name: str = None):
    """Track execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start
            stage_name = name or func.__name__
            self.ctx.config.metric_logger.log_timing(stage_name, elapsed)
            return result
        return wrapper
    return decorator

def notify_on_error(handler: Callable[[Exception, Any], None]):
    """Handle errors with notification."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                handler(e, self.ctx)
                raise
        return wrapper
    return decorator

def track_metric(metric_name: str, extractor: Callable[[Any], float]):
    """Track a metric from result."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            value = extractor(result)
            self.ctx.config.metric_logger.log_metric(metric_name, value)
            return result
        return wrapper
    return decorator

def log_if(condition: Callable[[Any], bool], message: str):
    """Conditionally log."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if condition(self.ctx):
                self.ctx.config.monitor.log(message.format(
                    result=result, 
                    ctx=self.ctx
                ))
            return result
        return wrapper
    return decorator
```

## Context Management

### Context Update Patterns

```python
# Context updates happen in three places:

# 1. I/O operations (explicit)
.io("update_budget", lambda result, ctx: ctx.update(
    remaining_budget=ctx.remaining_budget - result.cost
))

# 2. Control flow operations (automatic)
.do_while(
    lambda ctx: ctx.current_trial_idx < ctx.max_trials,
    inner_pipeline
)  # Automatically updates ctx.current_trial_idx

# 3. Never in stages (forbidden)
.stage("process", lambda data: data)  # Cannot mutate context
```

### Resource Access Pattern

```python
class SweepExperiment:
    def __init__(self, config: SweepExperimentConfig):
        self.cfg = config  # Direct reference to config
        self.ctx = SweepExperimentCtx(config)  # Context with mutable state
    
    # Methods access resources through self.cfg or self.ctx.config
    def suggest(self):
        return self.ctx.config.optimizer.suggest()
    
    def save_trial(self, result):
        return self.cfg.db_client.save_trial(  # Can use self.cfg directly
            self.cfg.sweep_name,
            self.ctx.current_trial_idx,  # Mutable state from context
            result
        )

# Clean separation:
# - self.cfg for immutable config and services
# - self.ctx for mutable flow state
# - self.ctx.config also references the same config
```

## Complete Example

```python
class SweepExperiment:
    """Complete sweep implementation with all patterns."""
    
    def __init__(self, config: SweepExperimentConfig):
        self.cfg = config
        self.ctx = SweepExperimentCtx(config)
    
    # ============= I/O Operations =============
    
    @log_before("Registering sweep in database")
    @log_after(lambda res, ctx: f"Registered with ID: {res.sweep_id}")
    def register_sweep(self):
        return self.cfg.db_client.register_sweep(self.cfg.sweep_name)
    
    @timed("history_load")
    def load_history(self):
        observations = self.cfg.db_client.fetch_observations(self.cfg.sweep_name)
        self.cfg.optimizer.load_observations(observations)
        return observations
    
    @timed("suggestion_generation")
    @log_after(lambda sugg, ctx: f"Trial {ctx.current_trial_idx}: {sugg.params}")
    def suggest(self):
        return self.cfg.optimizer.suggest()
    
    def observe_result(self, eval_result):
        return self.cfg.optimizer.observe(
            self.ctx.history[-1].suggestion,
            eval_result.score,
            eval_result.cost
        )
    
    @notify_on_error(lambda e, ctx: ctx.config.monitor.alert(f"Save failed: {e}"))
    def save_trial(self, eval_result):
        return self.cfg.db_client.save_trial(
            self.cfg.sweep_name,
            self.ctx.current_trial_idx,
            eval_result
        )
    
    # ============= Stage Operations =============
    
    @log_before(lambda ctx: f"Configuring trial {ctx.current_trial_idx}")
    def prepare_config(self, suggestion):
        return merge_configs(self.cfg.train_base_cfg, suggestion)
    
    @timed("model_training")
    @notify_on_error(lambda e, ctx: ctx.config.monitor.alert(f"Training failed: {e}"))
    def train_model(self, config):
        return train(config, seed=self.cfg.random_seed)
    
    @log_after(lambda res, ctx: 
        f"Score: {res.score:.4f}, Cost: ${res.cost:.2f}, Gap: {res.score - ctx.best_score:.4f}"
    )
    @track_metric("eval_score", lambda res: res.score)
    @track_metric("eval_cost", lambda res: res.cost)
    @log_if(lambda ctx: ctx.current_trial_idx % 10 == 0, 
            "Checkpoint at trial {ctx.current_trial_idx}")
    def evaluate_model(self, train_result):
        return evaluate_model(train_result.model, self.cfg.eval_base_cfg)
    
    # ============= Helper Methods =============
    
    def should_continue(self, ctx):
        return (
            ctx.current_trial_idx < ctx.max_trials and
            ctx.remaining_budget > 0 and
            not ctx.converged
        )
    
    def update_context(self, eval_result, ctx):
        return ctx.update(
            current_trial_idx=ctx.current_trial_idx + 1,
            remaining_budget=ctx.remaining_budget - eval_result.cost,
            best_score=max(ctx.best_score, eval_result.score),
            converged=eval_result.score >= ctx.convergence_threshold
        )
    
    # ============= Pipeline Builder =============
    
    def build_pipeline(self) -> Pipeline:
        """Compose the complete sweep pipeline."""
        return (
            Pipeline(self.ctx)
            
            # ===== Setup Phase =====
            .io("register", self.register_sweep)
            .io("load_history", self.load_history)
            .through(list[Observation])
            
            # ===== Main Sweep Loop =====
            .do_while(
                self.should_continue,
                
                Pipeline()
                    # Suggestion Generation
                    .io("suggest", self.suggest)
                    .through(Suggestion)
                    
                    # Configuration
                    .stage("prepare", self.prepare_config)
                    .through(TrainConfig)
                    
                    # Training
                    .stage("train", self.train_model)
                    .through(TrainResult)
                    
                    # Evaluation
                    .stage("evaluate", self.evaluate_model)
                    .through(EvalResult)
                    
                    # Recording
                    .io("observe", self.observe_result)
                    .io("save", self.save_trial)
                    
                    # Context Update
                    .io("update_context", self.update_context)
            )
            
            # ===== Finalization Phase =====
            .io("finalize", lambda: self.cfg.db_client.mark_complete(self.cfg.sweep_name))
            .hook(lambda: self.cfg.monitor.log(
                f"Sweep complete: {self.ctx.current_trial_idx} trials, "
                f"best score: {self.ctx.best_score:.4f}"
            ))
        )
    
    def run(self):
        """Execute the sweep experiment."""
        pipeline = self.build_pipeline()
        return pipeline.run()  # No arguments needed!

# ============= Usage =============

def create_sweep_config() -> SweepExperimentConfig:
    """Factory function to create configuration."""
    return SweepExperimentConfig(
        train_base_cfg=TrainConfig(
            learning_rate=1e-3,
            batch_size=32,
            epochs=100
        ),
        eval_base_cfg=EvalConfig(
            metrics=["accuracy", "f1_score"],
            test_split=0.2
        ),
        optimizer=ProteinOptimizer(
            config=ProteinConfig(
                method="bayes",
                acquisition_fn="ucb"
            )
        ),
        db_client=PostgresClient(connection_string="..."),
        metric_logger=WandbLogger(project="sweeps"),
        monitor=SlackMonitor(webhook_url="..."),
        sweep_name="awesome_sweep_2024",
        max_trials=100,
        budget=10000.0,
        convergence_threshold=0.95
    )

# Run the sweep
config = create_sweep_config()
experiment = SweepExperiment(config)
results = experiment.run()
```

## Migration Plan

### Phase 1: Core tAXIOM Enhancements
1. Implement enhanced Pipeline class with `io()` method
2. Add control flow operations (`do_while`, `do_until`)
3. Implement type checking at membranes
4. Add hook system

### Phase 2: Sweep Infrastructure
1. Define Protocol interfaces for services
2. Create base ExperimentCtx class
3. Implement decorator-based hooks
4. Create SweepExperimentConfig structure

### Phase 3: Migration
1. Refactor existing `sweep.py` to use new pattern
2. Update tests
3. Create migration guide for existing sweeps
4. Update documentation

### Phase 4: Optimization
1. Add caching for idempotent operations
2. Implement parallel execution where possible
3. Add retry logic for I/O operations
4. Create sweep resume capability

## Implementation Roadmap

### Week 1: Core Framework
- [ ] Enhance Pipeline with `io()` and `hook()` methods
- [ ] Implement control flow operations
- [ ] Add context management system
- [ ] Create type validation at membranes

### Week 2: Sweep Components
- [ ] Define service protocols
- [ ] Implement configuration/context classes
- [ ] Create decorator-based hooks
- [ ] Build experiment base class

### Week 3: Integration
- [ ] Migrate existing sweep to new pattern
- [ ] Update PROTEIN integration
- [ ] Integrate with wandb/cogweb
- [ ] Add comprehensive tests

### Week 4: Polish
- [ ] Performance optimization
- [ ] Documentation
- [ ] Example sweeps
- [ ] Migration guide

## Benefits Summary

1. **Explicit Effects**: Clear distinction between pure computation and I/O
2. **Type Safety**: Contracts enforced at stage boundaries
3. **Testability**: Each method independently testable
4. **Maintainability**: Clean separation of concerns
5. **Observability**: Hooks provide visibility without pollution
6. **Flexibility**: Decorator and inline hooks can be combined
7. **IDE Support**: Full autocomplete and type checking
8. **Reproducibility**: Deterministic stages with explicit randomness

This design provides a solid foundation for reliable, maintainable sweep experiments while maintaining the flexibility needed for research iteration.