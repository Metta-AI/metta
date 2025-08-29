# Sweep Infrastructure: Path to 10/10

## Current State: 7/10
A solid foundation with good architectural choices but needs implementation work before production use.

### Current Strengths
1. **Excellent Architecture** - Stateless, protocol-based design with clear separation of concerns
2. **Fault Tolerance** - Can be killed and restarted without losing progress
3. **Simplicity** - Synchronous operations with retry logic, easy to debug
4. **Extensibility** - Protocol-based design makes swapping implementations easy
5. **Resource Management** - Proper max_parallel_jobs enforcement and process reaping

### Current Weaknesses
1. **Missing Core Components** - No Store, Optimizer implementations
2. **Error Handling** - Limited error recovery and job retry mechanisms
3. **Observability** - Minimal metrics and monitoring
4. **Testing** - No test coverage
5. **Configuration Management** - Incomplete SweepConfig integration

---

## Path to 10/10: After Store & Optimizer Implementation

## 1. Robustness & Reliability (8→9)

### Comprehensive Error Handling
```python
@dataclass
class JobDefinition:
    # Add retry support
    max_retries: int = 3
    retry_count: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    retry_delay_seconds: int = 60
    
class SweepController:
    def _handle_failed_job(self, run: RunInfo):
        """Intelligent retry logic with exponential backoff"""
        if run.retry_count < run.max_retries:
            delay = run.retry_delay_seconds * (2 ** run.retry_count)
            self._requeue_with_backoff(run, delay)
            logger.info(f"Requeueing {run.run_id} after {delay}s (attempt {run.retry_count + 1})")
        else:
            self._mark_permanently_failed(run)
            self._notify_failure(run)
    
    def _requeue_with_backoff(self, run: RunInfo, delay: int):
        """Re-add job to queue with delay"""
        # Implementation here
    
    def _notify_failure(self, run: RunInfo):
        """Send notifications for permanent failures"""
        # Slack, email, PagerDuty integration
```

### Graceful Degradation
- **Fallback Dispatchers**: Try secondary dispatcher when primary fails
- **Partial Result Recovery**: Save intermediate results even if job crashes
- **Checkpoint Resume**: Resume from any checkpoint, not just latest
- **Store Resilience**: Handle Store disconnections with local queue

### State Validation
```python
class StateValidator:
    """Detect and fix inconsistent states"""
    
    def validate_run_state(self, run: RunInfo) -> list[str]:
        """Check for state inconsistencies"""
        issues = []
        if run.has_been_evaluated and not run.has_completed_training:
            issues.append("Evaluated without completing training")
        if run.completed_at and not run.started_at:
            issues.append("Completed without start time")
        return issues
    
    def auto_fix_states(self, run: RunInfo) -> RunInfo:
        """Attempt to fix common inconsistencies"""
        # Implementation here
```

---

## 2. Observability & Debugging (9→10)

### Rich Telemetry
```python
@dataclass
class RunMetrics:
    """Detailed metrics for each run"""
    # Timing metrics
    queue_time: float
    startup_time: float
    training_time: float
    eval_time: float
    total_time: float
    
    # Resource metrics
    gpu_utilization: list[float]
    memory_peak_gb: float
    disk_io_gb: float
    network_io_gb: float
    
    # Training metrics
    steps_per_second: float
    samples_processed: int
    checkpoint_size_mb: float
    
    def get_bottlenecks(self) -> list[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        if self.gpu_utilization and max(self.gpu_utilization) < 0.5:
            bottlenecks.append("Low GPU utilization")
        if self.queue_time > self.training_time * 0.1:
            bottlenecks.append("High queue time")
        return bottlenecks
```

### Event Stream
```python
class EventLog:
    """Structured event logging for post-mortem analysis"""
    
    def log_state_transition(self, 
                            run_id: str, 
                            from_state: JobStatus, 
                            to_state: JobStatus, 
                            metadata: dict):
        """Record all state changes with context"""
        event = {
            "timestamp": datetime.now(),
            "run_id": run_id,
            "transition": f"{from_state} -> {to_state}",
            "metadata": metadata,
            "stack_trace": traceback.format_stack()
        }
        self._append_event(event)
    
    def log_scheduling_decision(self, 
                               scheduler: str, 
                               decision: str, 
                               reasoning: dict):
        """Record why scheduler made specific decisions"""
        # Useful for debugging scheduling logic
    
    def log_optimization_step(self, 
                             suggestion: dict, 
                             expected_improvement: float,
                             acquisition_value: float):
        """Track optimizer's decision process"""
        # Helps understand hyperparameter selection
    
    def replay_events(self, sweep_id: str) -> Iterator[Event]:
        """Replay all events for debugging"""
        # Time-travel debugging capability
```

### Debug Mode
```python
class DebugController(SweepController):
    """Enhanced controller for debugging"""
    
    def dry_run(self, steps: int = 10) -> list[JobDefinition]:
        """Preview what jobs would be scheduled without executing"""
        simulated_jobs = []
        for _ in range(steps):
            jobs = self.scheduler.schedule(...)
            simulated_jobs.extend(jobs)
            # Simulate job completion
        return simulated_jobs
    
    def step_through(self):
        """Interactive debugging with breakpoints"""
        while True:
            input("Press Enter to continue to next scheduling decision...")
            # Execute one iteration
            self._run_single_iteration()
            self._print_state_summary()
```

---

## 3. Performance & Efficiency (9→10)

### Intelligent Scheduling
```python
class AdaptiveScheduler:
    """Adjusts strategy based on sweep progress"""
    
    def detect_convergence(self, observations: list[Observation]) -> bool:
        """Early stopping when improvement plateaus"""
        if len(observations) < 10:
            return False
        
        recent_scores = [obs.score for obs in observations[-10:]]
        improvement = max(recent_scores) - min(recent_scores)
        return improvement < self.convergence_threshold
    
    def adjust_exploration(self, phase: str):
        """Dynamically balance exploration/exploitation"""
        if phase == "early":
            self.exploration_rate = 0.8  # More exploration
        elif phase == "convergence":
            self.exploration_rate = 0.2  # More exploitation
    
    def predict_completion_time(self, 
                               completed: int, 
                               total: int, 
                               avg_duration: float) -> datetime:
        """ETA based on historical job duration"""
        remaining = total - completed
        eta_seconds = remaining * avg_duration
        return datetime.now() + timedelta(seconds=eta_seconds)
```

### Resource Optimization
```python
class ResourceOptimizer:
    """Maximize resource utilization"""
    
    def pack_jobs(self, jobs: list[JobDefinition]) -> list[JobBundle]:
        """Pack multiple small jobs on single GPU"""
        bundles = []
        current_bundle = JobBundle()
        
        for job in sorted(jobs, key=lambda j: j.estimated_memory):
            if current_bundle.can_fit(job):
                current_bundle.add(job)
            else:
                bundles.append(current_bundle)
                current_bundle = JobBundle(job)
        
        return bundles
    
    def use_spot_instances(self, job: JobDefinition) -> bool:
        """Determine if job is suitable for spot/preemptible instances"""
        return (job.max_retries > 2 and 
                job.checkpoint_interval < 300 and
                job.estimated_duration < 3600)
    
    def auto_scale(self, queue_depth: int, avg_wait_time: float) -> int:
        """Determine optimal number of workers"""
        if avg_wait_time > self.target_wait_time:
            return min(queue_depth, self.max_workers)
        return self.min_workers
```

### Cost Tracking
```python
@dataclass
class CostTracker:
    """Track and enforce budgets"""
    
    def estimate_job_cost(self, job: JobDefinition) -> float:
        """Estimate cost before running"""
        gpu_hours = job.estimated_duration / 3600 * job.gpus
        return gpu_hours * self.gpu_hour_rate
    
    def enforce_budget(self, remaining_budget: float, jobs: list[JobDefinition]) -> list[JobDefinition]:
        """Only schedule jobs that fit in budget"""
        affordable_jobs = []
        for job in jobs:
            cost = self.estimate_job_cost(job)
            if cost <= remaining_budget:
                affordable_jobs.append(job)
                remaining_budget -= cost
        return affordable_jobs
```

---

## 4. Developer Experience (9→10)

### Sweep DSL
```python
from metta.sweep import sweep, LogUniform, Choice, Range

@sweep(
    search_space={
        "lr": LogUniform(1e-4, 1e-1),
        "batch_size": Choice([32, 64, 128]),
        "dropout": Range(0.1, 0.5, step=0.1),
        "architecture": Choice(["resnet", "efficientnet", "vit"]),
    },
    strategy="bayesian",  # or "grid", "random", "asha", "pbt"
    max_parallel=10,
    budget_hours=100,
    metric="validation_accuracy",
    direction="maximize",
)
def train_model(config):
    """User's training function - automatically swept"""
    model = create_model(config["architecture"], dropout=config["dropout"])
    optimizer = Adam(lr=config["lr"])
    
    for epoch in range(100):
        # Training loop
        accuracy = train_epoch(model, optimizer, batch_size=config["batch_size"])
        
        # Report intermediate results for early stopping
        report_metrics({"validation_accuracy": accuracy, "epoch": epoch})
    
    return {"validation_accuracy": accuracy}
```

### Interactive CLI
```bash
# Live monitoring
metta sweep status --watch               # Live updating dashboard
metta sweep status --tail 46g3f         # Follow specific run logs

# Analysis
metta sweep analyze <sweep_id>          # Generate analysis report
metta sweep compare <id1> <id2>         # Compare two sweeps
metta sweep best <sweep_id> --top 5     # Get best configurations

# Control
metta sweep pause <sweep_id>            # Pause sweep execution
metta sweep resume <sweep_id>           # Resume paused sweep
metta sweep replay <sweep_id>           # Replay for debugging
metta sweep rollback <sweep_id> --jobs 5  # Undo last 5 jobs

# Export/Import
metta sweep export <sweep_id> --format yaml
metta sweep import sweep_config.yaml
```

### Smart Defaults
```python
class SmartDefaults:
    """Intelligent default configuration"""
    
    def detect_optimal_parallelism(self) -> int:
        """Auto-detect based on available resources"""
        gpus = torch.cuda.device_count()
        cpu_cores = os.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1e9
        
        # Heuristic based on resources
        if gpus > 0:
            return gpus * 2  # Oversubscribe slightly
        else:
            return min(cpu_cores // 4, int(memory_gb // 8))
    
    def infer_metric(self, code: str) -> str:
        """Detect what metric to optimize from code"""
        # Parse AST to find metric logging calls
        if "accuracy" in code:
            return "accuracy"
        elif "loss" in code:
            return "loss"
        return "objective"
    
    def suggest_search_space(self, model_type: str) -> dict:
        """Provide good starting search spaces"""
        presets = {
            "transformer": {
                "lr": LogUniform(1e-5, 1e-3),
                "warmup_steps": Range(100, 1000),
                "dropout": Range(0.0, 0.3),
            },
            "cnn": {
                "lr": LogUniform(1e-4, 1e-2),
                "batch_size": Choice([16, 32, 64]),
                "weight_decay": LogUniform(1e-5, 1e-3),
            }
        }
        return presets.get(model_type, {})
```

---

## 5. Advanced Features (9→10)

### Multi-Objective Optimization
```python
class ParetoOptimizer:
    """Optimize multiple competing objectives"""
    
    def get_pareto_frontier(self, observations: list[Observation]) -> list[Observation]:
        """Return non-dominated solutions"""
        frontier = []
        for obs in observations:
            dominated = False
            for other in observations:
                if self.dominates(other, obs):
                    dominated = True
                    break
            if not dominated:
                frontier.append(obs)
        return frontier
    
    def dominates(self, a: Observation, b: Observation) -> bool:
        """Check if a dominates b in all objectives"""
        better_in_any = False
        for objective in self.objectives:
            if a.metrics[objective] < b.metrics[objective]:
                return False  # Worse in this objective
            if a.metrics[objective] > b.metrics[objective]:
                better_in_any = True
        return better_in_any
    
    def select_compromise(self, frontier: list[Observation]) -> Observation:
        """Select balanced solution from Pareto frontier"""
        # Use knee-point detection or user preferences
```

### Transfer Learning
```python
class TransferScheduler:
    """Leverage results from related sweeps"""
    
    def warm_start_from(self, previous_sweep_id: str):
        """Start with knowledge from previous sweep"""
        prior_observations = self.store.fetch_observations(previous_sweep_id)
        
        # Build surrogate model from prior data
        self.surrogate_model = self._fit_gaussian_process(prior_observations)
        
        # Use transfer learning for initial suggestions
        self.initial_suggestions = self._transfer_top_k(prior_observations, k=5)
    
    def adapt_search_space(self, source_space: dict, target_task: str) -> dict:
        """Adapt hyperparameter ranges based on task similarity"""
        similarity = self._compute_task_similarity(source_task, target_task)
        
        adapted_space = {}
        for param, range in source_space.items():
            if similarity > 0.8:
                adapted_space[param] = range  # Use as-is
            else:
                # Expand range for different tasks
                adapted_space[param] = self._expand_range(range, factor=2.0)
        
        return adapted_space
```

### Distributed Coordination
```python
class FederatedSweep:
    """Coordinate sweeps across multiple clusters"""
    
    def __init__(self, sites: list[str]):
        self.sites = sites
        self.coordinators = {site: RemoteCoordinator(site) for site in sites}
    
    def aggregate_observations(self) -> list[Observation]:
        """Combine results from multiple sites"""
        all_observations = []
        for site, coordinator in self.coordinators.items():
            observations = coordinator.fetch_observations()
            # Add site metadata
            for obs in observations:
                obs.metadata["site"] = site
            all_observations.extend(observations)
        return all_observations
    
    def distribute_jobs(self, jobs: list[JobDefinition]) -> dict[str, list[JobDefinition]]:
        """Distribute jobs across sites based on capacity"""
        site_capacity = {site: coord.get_capacity() 
                        for site, coord in self.coordinators.items()}
        
        distribution = {site: [] for site in self.sites}
        for job in jobs:
            # Assign to site with most capacity
            best_site = max(site_capacity.items(), key=lambda x: x[1])[0]
            distribution[best_site].append(job)
            site_capacity[best_site] -= job.resource_requirements()
        
        return distribution
```

---

## 6. Production Excellence (9→10)

### Compliance & Auditing
```python
class AuditLog:
    """Immutable record of all decisions"""
    
    def __init__(self):
        self.events = []  # Append-only log
        self.merkle_tree = MerkleTree()  # For tamper detection
    
    def get_decision_trace(self, run_id: str) -> DecisionChain:
        """Show complete decision chain for a run"""
        events = [e for e in self.events if e.run_id == run_id]
        
        chain = DecisionChain()
        for event in events:
            chain.add_decision(
                timestamp=event.timestamp,
                decision_type=event.type,
                reasoning=event.metadata.get("reasoning"),
                alternatives=event.metadata.get("alternatives_considered"),
            )
        
        return chain
    
    def verify_reproducibility(self, sweep_id: str) -> bool:
        """Confirm sweep can be exactly reproduced"""
        original_events = self.get_sweep_events(sweep_id)
        
        # Replay with same random seeds
        replayed_events = self.replay_sweep(sweep_id)
        
        return self._events_match(original_events, replayed_events)
    
    def export_for_compliance(self, sweep_id: str) -> ComplianceReport:
        """Generate report for regulatory compliance"""
        return ComplianceReport(
            sweep_id=sweep_id,
            data_lineage=self.get_data_lineage(sweep_id),
            model_decisions=self.get_decision_trace(sweep_id),
            resource_usage=self.get_resource_usage(sweep_id),
            audit_hash=self.merkle_tree.get_root_hash(),
        )
```

### Integration Ecosystem
```python
class IntegrationHub:
    """Connect with external tools and services"""
    
    def export_to_prometheus(self, metrics: dict):
        """Export metrics for Grafana dashboards"""
        for name, value in metrics.items():
            self.prometheus_gauge[name].set(value)
    
    def notify_slack(self, event: str, details: dict):
        """Send notifications to Slack"""
        self.slack_client.post_message(
            channel="#ml-sweeps",
            text=f"Sweep Event: {event}",
            attachments=[self._format_details(details)]
        )
    
    def sync_with_mlflow(self, run: RunInfo):
        """Sync results with MLflow tracking"""
        with mlflow.start_run(run_id=run.run_id):
            mlflow.log_params(run.config)
            mlflow.log_metrics(run.metrics)
            mlflow.log_artifacts(run.artifact_path)
    
    def export_to_wandb(self, sweep: SweepMetadata):
        """Export sweep to Weights & Biases"""
        wandb.init(project="sweeps", name=sweep.sweep_id)
        wandb.log(sweep.to_metrics_dict())
```

### Safety Features
```python
class SafetyController:
    """Prevent catastrophic failures"""
    
    def detect_regression(self, new_score: float, baseline: float) -> bool:
        """Detect performance regression"""
        regression_threshold = 0.1  # 10% drop
        return new_score < baseline * (1 - regression_threshold)
    
    def canary_deployment(self, job: JobDefinition) -> bool:
        """Test risky configurations safely"""
        if self.is_risky(job):
            # Run with limited resources first
            canary_job = self.create_canary(job, scale=0.1)
            result = self.dispatcher.dispatch(canary_job)
            
            if result.success:
                return True  # Safe to run full job
            else:
                logger.warning(f"Canary failed for {job.run_id}")
                return False
    
    def enforce_quotas(self, user: str, resources: dict) -> bool:
        """Enforce resource quotas per user/team"""
        used = self.get_usage(user)
        quota = self.get_quota(user)
        
        for resource, amount in resources.items():
            if used[resource] + amount > quota[resource]:
                return False
        return True
    
    def detect_deadlock(self, jobs: list[RunInfo]) -> list[str]:
        """Detect circular dependencies or deadlocks"""
        # Build dependency graph
        graph = self.build_dependency_graph(jobs)
        
        # Find cycles
        cycles = self.find_cycles(graph)
        
        if cycles:
            return [f"Deadlock detected: {cycle}" for cycle in cycles]
        return []
```

---

## 7. Code Quality (9→10)

### Comprehensive Testing
```python
# Unit tests with property-based testing
from hypothesis import given, strategies as st

class TestScheduler:
    @given(
        observations=st.lists(
            st.builds(Observation, 
                     score=st.floats(0, 1),
                     cost=st.floats(0, 1000))
        )
    )
    def test_scheduler_invariants(self, observations):
        """Verify scheduler always makes valid decisions"""
        scheduler = SequentialScheduler()
        jobs = scheduler.schedule(metadata, observations)
        
        # Invariants
        assert all(isinstance(job, JobDefinition) for job in jobs)
        assert len(jobs) <= scheduler.max_parallel
        assert all(job.run_id for job in jobs)
    
    def test_failure_recovery(self):
        """Test recovery from various failure modes"""
        scenarios = [
            "store_disconnect",
            "dispatcher_timeout",
            "corrupt_state",
            "oom_error",
        ]
        
        for scenario in scenarios:
            with self.simulate_failure(scenario):
                controller = SweepController(...)
                controller.run()
                # Should recover gracefully
                assert controller.is_healthy()
```

### Performance Benchmarks
```python
class SweepBenchmark:
    """Benchmark suite for regression detection"""
    
    def measure_scheduling_latency(self, num_observations: int) -> float:
        """Measure time to make scheduling decision"""
        observations = self.generate_observations(num_observations)
        
        start = time.perf_counter()
        scheduler.schedule(metadata, observations)
        end = time.perf_counter()
        
        return end - start
    
    def measure_store_throughput(self) -> float:
        """Measure Store operations per second"""
        operations = 1000
        
        start = time.perf_counter()
        for _ in range(operations):
            store.update_run_summary(run_id, {"metric": random.random()})
        end = time.perf_counter()
        
        return operations / (end - start)
    
    def measure_memory_usage(self, num_runs: int) -> float:
        """Measure memory scaling with number of runs"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Create many runs
        runs = [self.create_run() for _ in range(num_runs)]
        controller = SweepController(...)
        controller._compute_metadata_from_runs(runs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return peak / 1024 / 1024  # MB
    
    def generate_report(self) -> BenchmarkReport:
        """Run all benchmarks and generate report"""
        return BenchmarkReport(
            scheduling_latency=self.measure_scheduling_latency(1000),
            store_throughput=self.measure_store_throughput(),
            memory_usage=self.measure_memory_usage(10000),
            timestamp=datetime.now()
        )
```

### Documentation
```python
class SweepDocGenerator:
    """Auto-generate documentation from code"""
    
    def generate_scheduler_docs(self, scheduler_class):
        """Extract docstrings and create markdown docs"""
        docs = f"# {scheduler_class.__name__}\n\n"
        docs += f"{scheduler_class.__doc__}\n\n"
        
        docs += "## Methods\n\n"
        for method in inspect.getmembers(scheduler_class, predicate=inspect.ismethod):
            docs += f"### {method.__name__}\n"
            docs += f"{method.__doc__}\n\n"
        
        return docs
    
    def generate_example_notebook(self, sweep_config: dict) -> str:
        """Create Jupyter notebook with examples"""
        notebook = {
            "cells": [
                self.create_markdown_cell("# Sweep Example"),
                self.create_code_cell(self.generate_setup_code()),
                self.create_code_cell(self.generate_sweep_code(sweep_config)),
                self.create_code_cell(self.generate_analysis_code()),
            ]
        }
        return json.dumps(notebook)
```

---

## Summary: The 10/10 System

A 10/10 sweep infrastructure would have:

### Core Qualities
1. **Zero Surprises** - Predictable behavior, excellent error messages, clear documentation
2. **Self-Healing** - Automatic recovery from failures, retry logic, state validation
3. **Observable** - Rich telemetry, event logging, real-time monitoring
4. **Efficient** - Resource optimization, cost tracking, intelligent scheduling
5. **Delightful** - Great CLI, smart defaults, minimal configuration needed
6. **Production-Ready** - Battle-tested, compliant, integrated with existing tools
7. **Extensible** - Easy to add new schedulers/optimizers/dispatchers
8. **Fast** - Subsecond scheduling, minimal overhead, efficient resource use

### Key Differentiators
- **Adaptive Intelligence**: System learns and improves over time
- **Transfer Learning**: Leverages knowledge from previous sweeps
- **Multi-Objective**: Handles competing objectives elegantly
- **Federated**: Can coordinate across multiple sites/clusters
- **Compliant**: Full audit trail and reproducibility
- **Safe**: Prevents regressions and resource overuse

### Implementation Priority
1. **Phase 1** (Current): Basic Store, Optimizer, error handling
2. **Phase 2**: Rich telemetry, smart scheduling, resource optimization
3. **Phase 3**: Advanced features (transfer learning, multi-objective)
4. **Phase 4**: Production excellence (compliance, safety, integration)

The current architecture is an excellent foundation. These enhancements would make it world-class, comparable to or better than systems at top ML companies. The key is iterating based on real usage patterns rather than building everything upfront.