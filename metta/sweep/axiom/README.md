# tAXIOM

```
        _      _____  __  __  _____   ____   __  __ 
       | |_   |  _  | \ \/ / |_   _| / __ \ |  \/  |
       | __|  | |_| |  \  /    | |  | |  | || |\/| |
       | |_   |  _  |  /  \    | |  | |__| || |  | |
        \__|  |_| |_| /_/\_\  |___|  \____/ |_|  |_|
                                                     
```

> **Portable experiment specifications for reproducible RL research**

Most RL results are hard to compare. Training loops are bespoke; side-effects get tangled with compute; "pipelines" are often just a for-loop with logging sprinkled in. **tAXIOM** solves two critical problems:

1. **Reproducible comparison** - Different teams can run identical experimental protocols with their own implementations
2. **Systematic experimentation** - A/B testing, ablation studies, and hyperparameter sweeps become trivial when it's completely clear, by definition, what is changing and what is static

The core insight: express complete experiments—including their control flow—as explicit, shareable pipelines. When you swap a `.join()` block or modify a stage, you know *exactly* what changed. Nothing is hidden.

---

## The Core Model

* **Pipelines** compose **pure computation** (stages) and **explicit side-effects** (I/O)
* **Contracts** (checks) and **observers** (hooks) at boundaries keep runs honest and debuggable  
* **Black-box subsystems** via `.join()` let collaborators swap implementations (PyTorch/RLlib/PufferLib)
* **Minimal control flow** (`.repeat()`, `.branch()`, `.map()`) expresses complete experiments without external orchestration

This creates shareable experimental protocols where the entire experiment—including iterations and conditions—is visible in one pipeline.

---

## Quick Example: Complete PBT Pipeline

**A full Population-Based Training experiment as a single, shareable pipeline:**

```python
def make_pbt_pipeline(spec) -> Pipeline:
    # Spec provides the implementations directly (preferred approach)
    return (
        Pipeline()
        # Main experiment loop
        .repeat("generations",
                body=(Pipeline()
                      # Train/eval each individual in population
                      .map("per_individual", key="pop",
                           body=(Pipeline()
                                 .join("train_block", sub=spec.train_block(), 
                                       propagate_global_checks=True)
                                 .through(checks={"score_present": FAIL(required_keys(["score"]))})
                                 .join("eval_block", sub=spec.eval_block() or Pipeline(), 
                                       propagate_global_checks=True)
                                 .logf("Gen {payload.gen}: Individual {payload.item.id} → {payload.score:.3f}")),
                           gather="list", opaque=True)
                      # Evolve population
                      .join("select_mutate_block", sub=spec.select_mutate_block(),
                            propagate_global_checks=True)
                      .through(checks={"pop_valid": FAIL(required_keys(["pop"]))})),
                max_steps=spec.max_generations,
                annotate="gen",
                until=lambda v: all(ind.get("score", 0) >= spec.target_score 
                                   for ind in v.get("pop", [])),
                opaque=True)
    )

# Run with spec that provides implementations
spec = ExperimentSpec(
    train_block=lambda: Pipeline().stage("train", torch_ppo_train),
    eval_block=lambda: Pipeline().stage("eval", evaluate_policy),  
    select_mutate_block=lambda: Pipeline()
        .stage("select", select_top_k)
        .stage("mutate", perturb_hyperparams),
    max_generations=100,
    target_score=0.85
)

pipeline = make_pbt_pipeline(spec)

# Run with initial population
value = {
    "pop": [{"id": i, "seed": s} for i, s in enumerate(spec.seeds)],
    "cfg": {"backend": spec.backend, "env": spec.env_name}
}
result = make_pbt_pipeline(spec).run(value, ctx)
```

**The entire PBT algorithm is now a portable pipeline** that different teams can run with their own training backends.

---

## Core API

### Basic Pipeline Operations

```python
from metta.sweep.axiom import Pipeline

# Preferred: Spec-first approach with direct binding
pipeline = (
    Pipeline()
    # Pure computation: value -> value
    .stage("compute", pure_function)
    
    # Side-effects allowed: (value, ctx) -> value  
    .io("fetch", fetch_from_api)
    
    # Contracts & observation at boundaries
    .through(
        checks={"score_valid": FAIL(required_keys(["score"])),      # Named checks
                "no_nans": WARN(no_nan(), sample_every=10, n_of_m=(5,20))},  # Statistical sampling
        hooks=[log_metrics, send_notification]                       # Observation only
    )
    
    # Black-box sub-pipelines with implementations from spec
    .join("train_block", sub=spec.train_block(), 
          propagate_global_checks=True)
    
    # Inline logging without lambdas
    .logf("Trial {payload.id}: score={payload.score:.3f}")
)

# Alternative: Standalone pipeline distribution (optional sugar)
standalone_pipeline = (
    Pipeline()
    .require_join("train_block", exit_checks=[FAIL(required_keys(["score"]))])
    .require_join("eval_block", exit_checks=[WARN(prob_simplex(1e-6))])
    # ... rest of pipeline
)
# Receiver fills the slots:
filled = (standalone_pipeline
    .provide_join("train_block", my_train_pipeline())
    .provide_join("eval_block", my_eval_pipeline())
)
```

### Control Flow Primitives

```python
# Repeat with bounds and pure conditions
pipeline.repeat(
    "generations",
    body=generation_pipeline,
    max_steps=100,                    # Required upper bound
    annotate="gen",                   # Adds {"gen": i} to value
    until=lambda v: v["score"] > 0.9, # Pure predicate on value
    opaque=True                       # Collapse iterations in render
)

# Conditional execution
pipeline.branch(
    "strategy_choice",
    cond=lambda v: v["epoch"] < 10,
    then=exploration_pipeline,
    else_=exploitation_pipeline,
    opaque=False                      # Show chosen branch in render
)

# Map over collections
pipeline.map(
    "evaluate_envs", 
    key="environments",               # Maps over value["environments"]
    body=eval_pipeline,               # Each item gets {"item": env, "cfg": ...}
    gather="list",                    # Collect results as list
    parallel=False,                   # Serial execution (deterministic)
    opaque=True                       # Collapse per-item details
)
```

### Key Design Principles

1. **Config flows in value**: `{"cfg": {...}, "data": ...}` makes pipelines portable
2. **Pure predicates**: All conditions operate on values, not context
3. **Mandatory bounds**: No infinite loops possible (`max_steps` required)
4. **Composable control flow**: Control flow primitives are just structured pipeline composition

---

## Experiment Specifications

Define experiments with **Pydantic** for validation and documentation:

```python
from pydantic import BaseModel, Field
from typing import Callable, Literal

class ExperimentSpec(BaseModel):
    # Swappable implementations
    train_block: Callable[[], Pipeline]
    eval_block: Callable[[], Pipeline] | None = None
    select_mutate_block: Callable[[], Pipeline]
    
    # Experiment parameters
    pop_size: int = 10
    max_generations: int = 100
    target_score: float = 0.85
    seeds: list[int] = Field(default_factory=lambda: list(range(5)))
    
    # Backend configuration
    backend: Literal["torch", "rllib", "pufferlib"] = "torch"
    env_name: str = "CartPole-v1"
    device: str = "cuda"

# Bind implementations
spec = ExperimentSpec(
    train_block=lambda: Pipeline().stage("train", torch_ppo_train),
    eval_block=lambda: Pipeline().stage("eval", evaluate_policy),
    select_mutate_block=lambda: Pipeline()
        .stage("select", select_top_k)
        .stage("mutate", perturb_hyperparams)
)

# Create and run the complete experiment pipeline
pipeline = make_pbt_pipeline(spec)
result = pipeline.run(initial_value, ctx)
```

Teams share the `ExperimentSpec` and pipeline structure. Each team provides their own `train_block`, `eval_block`, and `select_mutate_block` implementations.

---

## Checks: Statistical Process Control for RL

Checks are **on-the-fly control variables** that catch divergence without affecting results:

```python
# Simple invariants
FAIL(required_keys(["score", "artifact_uri"]))  # Must have these keys
WARN(no_nan())                                   # No NaN values
WARN(prob_simplex(1e-6))                        # Valid probability distribution

# Statistical sampling for noisy RL environments
WARN(grad_band(min=1e-6, max=100),              # Gradient magnitude bounds
     sample_every=10,                            # Check every 10th call
     n_of_m=(5, 20))                            # Warn if 5 of last 20 fail

# Named checks for better reporting in manifests
.through(checks={
    "score_valid": FAIL(required_keys(["score"])),
    "simplex": WARN(prob_simplex(1e-6)),
    "gradients_stable": WARN(grad_band(1e-6, 100), sample_every=10, n_of_m=(5,20))
})
```

With named checks, manifests report: `"Check 'gradients_stable' warned 3 times"` instead of `"Check #2 warned 3 times"`.

---

## Manifests: Reproducible Science

Every run produces a **manifest** - a complete reproduction recipe:

```yaml
experiment:
  spec: ExperimentSpec(...)           # Full configuration
  pipeline_hash: sha256:abc123...     # Code fingerprint
  joined_blocks:                      # What was bound to each slot
    train_block: torch_ppo_v2.3
    eval_block: standard_eval_v1.0
    
environment:
  python: 3.11.5
  torch: 2.1.0
  cuda: 12.1
  
contracts:
  checks_run: 847
  warnings: 12
  failures: 0
  
metrics:
  time_to_target: 3.2h               # Key comparisons
  samples_per_second: 45000
  final_score: 0.82
  
artifacts:
  checkpoints: wandb://project/run/model_final
  logs: s3://bucket/experiment/logs/
```

Share the manifest to enable exact reproduction and fair comparison.

---

## What Makes tAXIOM Different

### Perfect Clarity for A/B Testing and Ablations

When everything is explicit, controlled experimentation becomes trivial:

```python
# Define protocol with variable points
protocol = (Pipeline()
    .io("load_data", load_dataset)
    .require_join("augmentation")  # Variable A
    .stage("train", train_model)
    .require_join("optimizer")     # Variable B
    .io("evaluate", compute_metrics)
)

# A/B Test: Change ONLY augmentation
pipeline_a = protocol.provide_join("augmentation", aggressive_augment())
                     .provide_join("optimizer", baseline_optimizer())

pipeline_b = protocol.provide_join("augmentation", minimal_augment())
                     .provide_join("optimizer", baseline_optimizer())

# Ablation: Remove component entirely
pipeline_with = protocol.provide_join("curiosity", curiosity_bonus())
pipeline_without = protocol.provide_join("curiosity", Pipeline())  # No-op

# Interaction study: Test all combinations
for aug in [minimal, aggressive]:
    for opt in [sgd, adam]:
        pipeline = protocol.provide_join("augmentation", aug())
                           .provide_join("optimizer", opt())
```

You know **exactly** what changed because the pipeline definition IS the complete specification. No hidden state, no ambient configuration, no surprises.

See [AB_TESTING_EXAMPLE.md](AB_TESTING_EXAMPLE.md) for a complete walkthrough of testing competing hypotheses with scientific rigor.

### Spec-First Design with Pydantic Validation

Experiments are defined through **typed specifications** that ensure all implementations are provided upfront:

```python
class ExperimentSpec(BaseModel):
    train_block: Callable[[], Pipeline]      # Required
    eval_block: Callable[[], Pipeline] | None  # Optional
    # ... parameters, resources, etc.

# Implementations bound directly through spec
Pipeline().join("train_block", sub=spec.train_block())
```

This approach gives you Pydantic validation, IDE autocomplete, and clear contracts. For standalone pipeline distribution, `.require_join()`/`.provide_join()` offer optional ergonomic sugar.

### Minimal, Structured Control Flow

Unlike frameworks that become full programming languages, tAXIOM provides exactly three control flow primitives:

- **`.repeat()`** - Bounded iteration with pure convergence conditions
- **`.branch()`** - Conditional execution based on value predicates  
- **`.map()`** - Apply pipelines to collections

These are **composition operators**, not computation. They structure how pipelines combine while keeping all logic pure and deterministic.

### Separation of Concerns Is Enforceable

The framework provides exactly the tools needed to maintain clean separation:

- **Checks** verify invariants without changing outcomes
- **Hooks** provide uniform observability without affecting results
- **Guards** control execution (timeout, placement) without touching logic
- **Joins** enable modularity without coupling

### Everything Is Explicit

- **Config travels in values**: `{"cfg": {...}, "data": ...}` - no hidden state
- **Predicates are pure**: All decisions based on values, not ambient context
- **Bounds are mandatory**: Every `.repeat()` requires `max_steps` - no infinite loops
- **Errors surface locally**: Failures report exact pipeline paths like `generations#03/train_block/train`

This explicitness makes experiments reproducible and debuggable.

---

## Publishing Your Experiment

### Preferred: Spec-First Approach

Share experiments through typed specifications:

```
my_experiment/
├── spec.py             # Pydantic ExperimentSpec with slots
├── pipeline.py         # Pipeline using spec.train_block() etc.
├── examples/
│   ├── torch_ppo.py    # ExperimentSpec with PyTorch implementations
│   ├── rllib_ppo.py    # ExperimentSpec with RLlib implementations
│   └── pufferlib.py    # ExperimentSpec with PufferLib implementations
├── manifests/          # Results from published runs
│   └── baseline.yaml
└── README.md           # What to compare (TTT, SPS, cost)
```

Collaborators:
1. Create their own `ExperimentSpec` with their implementations
2. Run `make_pbt_pipeline(spec)` to get the pipeline
3. Execute and compare manifests

### Alternative: Standalone Pipeline Distribution

For shipping pipelines without Experiment scaffolding:

```python
# publisher.py - Pipeline with holes
pipeline = (Pipeline()
    .require_join("train_block", exit_checks=[...])
    .require_join("eval_block")
    # ... rest of pipeline
)

# receiver.py - Fill the holes
filled = (pipeline
    .provide_join("train_block", my_train())
    .provide_join("eval_block", my_eval())
)
```

Use this when you want maximum flexibility without requiring Pydantic specs.

---

## Installation & Usage

```bash
# Install tAXIOM
pip install metta-axiom

# Run with your backend
python examples/torch_ppo.py --seeds 1,2,3,4,5 --target 0.80

# Compare results
axiom compare manifests/baseline.yaml manifests/my_run.yaml
```

---

## Design Philosophy

### Separate Structure from Implementation

Traditional approach: "Here's my complete training code, try to reproduce it"
tAXIOM approach: "Here's the experimental structure, bring your own trainer"

This separation enables true apples-to-apples comparison across different implementations.

### Contracts Over Documentation

Instead of documenting "the score should be between 0 and 1", enforce it:

```python
.through(checks=[FAIL(lambda v: 0 <= v["score"] <= 1)])
```

Contracts are executable documentation that can't drift from reality.

### Pragmatism Over Purity

- A 500-line stage that works is better than 50 perfectly factored stages that don't
- Use the structure when it helps, ignore it when it doesn't
- The goal is reproducible science, not architectural beauty

### Beyond Experiments

While we developed tAXIOM as an experiment sharing paradigm, we believe it may be appropriate for use in production code (i.e. removed from the experimental feedback loop and into the infrastructural RL loop), especially in instances where said code changes quickly and could benefit from the clear separation of concerns during iteration.

---

## Roadmap

### Current Status: MVP Complete ✅
- Basic pipeline composition (stage, io, through)
- Hooks for observation
- Integration with existing sweep infrastructure
- Working hyperparameter optimization

### Next: Production Features

1. **Core Pipeline Features** (Week 1)
   - **`.join()`** with `sub` and `propagate_global_checks` parameters
   - **`.logf()`** - Inline logging without lambdas
   - **`opaque`** parameter for render control
   - **`.require_join()`/`.provide_join()`** - Optional sugar for standalone pipeline distribution

2. **Control Flow Primitives** (Week 1)
   - **`.repeat()`** with `max_steps`, `until`, `annotate`, `opaque`
   - **`.branch()`** with pure predicates and `opaque`
   - **`.map()`** with `gather`, `parallel=False`, `opaque`

3. **Checks & Hooks** (Week 1)
   - **Named checks** for clear manifest reporting
   - **Statistical sampling**: `sample_every`, `n_of_m`
   - Separate checks from hooks in `.through()`
   - Check library: `no_nan`, `prob_simplex`, `grad_band`, `required_keys`

4. **Experiment & Manifest Layer** (Week 2)
   - Pydantic-based ExperimentSpec
   - Comprehensive manifest generation:
     - Experiment snapshot (all parameters)
     - Pipeline fingerprint (code hash)
     - Joined implementations
     - Environment (Python/CUDA versions)
     - Check verdicts (WARN/FAIL counts by name)
     - Key metrics (TTT@target, SPS, cost)
     - Artifact URIs
   - Reference implementations (torch, rllib, pufferlib)

5. **Guards & Tooling**
   - Guard library: `@timeout`, `@retry`, `@gpu_required`, `@master_only`
   - **`axiom validate`** - Dry-run with checks but no compute
   - **`axiom compare`** - Diff manifests across runs
   - **`axiom replay`** - Reproduce from manifest
   - Error path reporting: `generations#03/train_block/train`

### Future: Advanced Features

**Endomorphic Composition** - For experimenting with compositional losses and transformations:

```python
# Compose different loss functions dynamically
pipeline = Pipeline().compose(
    "loss_computation",
    lambda v: {**v, "policy_loss": compute_policy_loss(v)},
    lambda v: {**v, "value_loss": compute_value_loss(v)},
    lambda v: {**v, "entropy_bonus": compute_entropy(v) * v["cfg"]["entropy_coef"]},
    lambda v: {**v, "total_loss": v["policy_loss"] + v["value_loss"] - v["entropy_bonus"]}
)

# Build loss pipelines from configuration
def make_loss_pipeline(loss_config):
    stages = []
    if loss_config.use_policy_loss:
        stages.append(add_policy_loss)
    if loss_config.use_value_loss:
        stages.append(add_value_loss)
    if loss_config.use_curiosity:
        stages.append(add_curiosity_loss)
    if loss_config.use_auxiliary_tasks:
        stages.extend([add_inverse_dynamics, add_forward_prediction])
    
    return Pipeline().compose("compute_loss", *stages)
```

This enables systematic ablation studies and loss recipe sharing - critical for RL research where loss engineering is often the key to breakthrough performance.

---

## FAQ: Anticipated Pushback (and Honest Answers)

**"Refactoring my loop into a pipeline seems like work."**

It is—and it mirrors the scientific method: isolate what changes, make boundaries explicit, and encode assumptions as checks. Once you've done it once, the map is shareable.

**"Do I have to use your Experiment class?"**

No. It's a pattern that plays beautifully with Pydantic and clarifies what's open vs. fixed. If you want to wire pipelines by hand, go for it.

**"Are we trying to do too much?"**

There's a predictable critique of frameworks like this: "Isn't this over-engineering?" Here's the honest answer:

We're not pushing tAXIOM as an all-in-one coding paradigm for ML/RL. Pipelines are modular, contract-bound units of computational logic. We orchestrate them however we want—outside the pipeline. That alone removes the heaviest design burden: control-flow syntax.

The remaining features—hooks, guards, checks—aren't "extras." They are the small, sharp tools that make separation of concerns enforceable in RL/ML:
- **Checks** are on-the-fly control variables. They can be trivial (same dtype) or critical (no NaNs, probability simplex), and they cannot change outcomes.
- **Hooks** give uniform observability and never affect results.
- **Guards** (timeouts, placement) wrap compute without touching it.

A platform that preaches a rigid separation of concerns is incomplete unless it also gives you the tools to enforce that separation. tAXIOM provides exactly those tools—no more, no less.

---

## Contributing

We welcome contributions that maintain tAXIOM's focus on **portable, comparable experiments**:

- Keep the core minimal - resist feature creep
- Maintain separation between pipelines and orchestration
- Ensure all checks are side-effect free
- Document with examples, not just API references

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT - See [LICENSE](LICENSE) for details.

---

*tAXIOM: Portable. Comparable. Reproducible.*