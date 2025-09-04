# Metta Sweep Infrastructure Analysis Report

## Executive Summary

The Metta sweep infrastructure is a sophisticated hyperparameter optimization (HPO) system designed to support various optimization methods including Bayesian Optimization, Evolution Strategies, and other state-of-the-art techniques. The architecture follows clean separation of concerns with protocol-based design, enabling flexibility and extensibility.

## Table of Contents

1. [High-Level Architecture Overview](#high-level-architecture-overview)
2. [Component-by-Component Analysis](#component-by-component-analysis)
3. [Workflow Analysis](#workflow-analysis)
4. [Areas of Excellence](#areas-of-excellence)
5. [Areas for Improvement](#areas-for-improvement)
6. [Failure Scenario Analysis](#failure-scenario-analysis)
7. [Retry Requirements](#retry-requirements)

---

## 1. High-Level Architecture Overview

### Core Design Principles

The sweep infrastructure operates on several key principles:

1. **Protocol-Based Architecture**: Components communicate through well-defined protocols (Scheduler, Store, Dispatcher, Optimizer)
2. **Stateless Design**: Most components are stateless, with state managed centrally
3. **Separation of Concerns**: Clear boundaries between orchestration, scheduling, optimization, and execution
4. **Resilience**: Built-in fault tolerance through in-memory state tracking and WandB persistence

### System Flow

```
User → SweepTool → orchestrate_sweep → SweepController → [Scheduler, Store, Dispatcher, Optimizer]
```

The system follows a control loop pattern where:
1. **SweepController** orchestrates the entire process
2. **Scheduler** decides what jobs to run
3. **Optimizer** suggests hyperparameters
4. **Store** persists state to WandB
5. **Dispatcher** executes jobs locally or on cloud

---

## 2. Component-by-Component Analysis

### 2.1 SweepTool (`metta/tools/sweep.py`)

**Purpose**: Entry point and configuration interface for sweeps

**Key Responsibilities**:
- Parse command-line arguments and configuration
- Initialize all components (Store, Dispatcher, Scheduler, Optimizer)
- Configure execution mode (local, cloud, hybrid)
- Generate sweep names and directories
- Provide final summary and reporting

**Strengths**:
- Clean configuration interface using Pydantic
- Flexible dispatcher configuration (LOCAL, SKYPILOT, HYBRID_REMOTE_TRAIN)
- Auto-configuration for WandB and stats server

### 2.2 SweepController (`metta/sweep/controller.py`)

**Purpose**: Central orchestrator managing the sweep lifecycle

**Key Responsibilities**:
- Main control loop execution
- In-memory state tracking (dispatched_trainings, dispatched_evals, completed_runs)
- Capacity management for parallel jobs
- Run status monitoring and metadata computation
- Job filtering and dispatch coordination

**Key Innovation**: 
The controller maintains authoritative in-memory state (`dispatched_trainings`, `dispatched_evals`, `completed_runs`) separate from WandB, solving API caching issues.

**Control Loop Steps**:
1. Fetch all runs from Store
2. Compute metadata and update completed runs
3. Display monitoring table
4. Get job schedule from Scheduler
5. Filter jobs by capacity and dispatch status
6. Execute jobs through Dispatcher
7. Update completed runs and check for eval completion
8. Sleep (5s if eval completed, otherwise monitoring_interval)

### 2.3 OptimizingScheduler (`metta/sweep/schedulers/optimizing.py`)

**Purpose**: Scheduling logic for job sequencing and evaluation

**Key Responsibilities**:
- Determine when to schedule training vs evaluation jobs
- Check for runs needing evaluation (TRAINING_DONE_NO_EVAL status)
- Enforce max_trials limit
- Wait for incomplete jobs before scheduling new ones
- Build job definitions with proper overrides

**Decision Flow**:
1. First priority: Schedule evaluations for completed training
2. Check max trials limit
3. Wait for incomplete jobs
4. Request new suggestions from Optimizer
5. Create job definitions with proper configuration

### 2.4 ProteinOptimizer (`metta/sweep/optimizer/protein.py`)

**Purpose**: Adapter for the Protein optimization library

**Key Responsibilities**:
- Interface between sweep system and Protein algorithms
- Support multiple optimization methods (Bayes, Random, Genetic)
- Convert observations to Protein format
- Generate hyperparameter suggestions

**Supported Methods**:
- **Bayesian Optimization**: GP-based with multiple acquisition functions (naive, EI, UCB)
- **Genetic Algorithm**: Pareto front evolution
- **Random Search**: Baseline random sampling

### 2.5 WandbStore (`metta/sweep/stores/wandb.py`)

**Purpose**: Persistence layer using Weights & Biases

**Key Responsibilities**:
- Initialize runs in WandB
- Fetch run information with status determination
- Update run summaries
- Convert WandB data to internal RunInfo format
- Handle observation storage

**Status Determination Logic**:
- Maps WandB states to internal JobStatus enum
- Checks training completion via timestep comparison
- Detects evaluation through metrics or flags

### 2.6 Dispatchers

#### LocalDispatcher (`metta/sweep/dispatcher/local.py`)
- Executes jobs as local subprocesses
- Captures and streams output
- Manages process lifecycle
- Returns PIDs for tracking

#### SkypilotDispatcher (`metta/sweep/dispatcher/skypilot.py`)
- Launches jobs on cloud resources
- Fire-and-forget execution model
- GPU and node configuration
- Returns UUIDs for tracking

#### RoutingDispatcher (`metta/sweep/dispatcher/routing.py`)
- Routes jobs based on type
- Enables hybrid execution modes
- Default: Training on cloud, evaluation locally

### 2.7 Models (`metta/sweep/models.py`)

**Key Data Models**:
- **JobDefinition**: Complete job specification
- **RunInfo**: Standardized run information
- **JobStatus**: Lifecycle state enum
- **SweepMetadata**: Aggregate sweep statistics
- **Observation**: HPO observation (score, cost, suggestion)

---

## 3. Workflow Analysis

### 3.1 Training Job Lifecycle

```
1. Scheduler requests suggestion from Optimizer
2. Controller creates WandB run (Store.init_run)
3. Controller dispatches job (Dispatcher.dispatch)
4. Job added to dispatched_trainings
5. Training executes (logs to WandB)
6. Status transitions: PENDING → IN_TRAINING → TRAINING_DONE_NO_EVAL
```

### 3.2 Evaluation Job Lifecycle

```
1. Scheduler detects TRAINING_DONE_NO_EVAL status
2. Controller dispatches evaluation job
3. Job added to dispatched_evals
4. Evaluation executes with policy_uri
5. Status transitions: TRAINING_DONE_NO_EVAL → IN_EVAL → EVAL_DONE_NOT_COMPLETED
6. Controller records observation
7. Status transitions: EVAL_DONE_NOT_COMPLETED → COMPLETED
```

### 3.3 Optimization Loop

```
1. Optimizer receives observations from completed runs
2. Bayesian: Train GP models on score and cost
3. Find Pareto optimal points
4. Sample candidates around Pareto centers
5. Apply acquisition function
6. Return best suggestion
```

---

## 4. Areas of Excellence

### 4.1 Robust State Management
- **In-memory dispatch tracking** prevents duplicate job submissions
- **Separation from WandB state** eliminates API caching issues
- **Capacity management** uses dispatched_trainings - completed_runs for accurate tracking

### 4.2 Clean Architecture
- **Protocol-based design** enables easy component replacement
- **Clear separation of concerns** between orchestration, scheduling, and execution
- **Stateless components** (Scheduler, Optimizer) improve testability

### 4.3 Flexible Execution Modes
- **Three dispatcher modes** (LOCAL, SKYPILOT, HYBRID_REMOTE_TRAIN)
- **Routing dispatcher** enables per-job-type execution strategies
- **Configurable GPU and node allocation**

### 4.4 Advanced Optimization Support
- **Multiple optimization methods** (Bayes, Genetic, Random)
- **Multiple acquisition functions** (naive, EI, UCB)
- **Pareto-optimal search** for multi-objective optimization
- **Cost-aware optimization** with max_suggestion_cost

### 4.5 Monitoring and Observability
- **Real-time status tables** with progress tracking
- **Comprehensive logging** at all levels
- **WandB integration** for metrics and artifacts
- **Final sweep summaries** with best results

### 4.6 Fault Tolerance
- **Graceful error handling** in control loop
- **Resume capability** through WandB persistence
- **Refractory period** after eval completion prevents race conditions

---

## 5. Areas for Improvement

### 5.1 Retry Mechanisms
**Issue**: Limited retry logic for transient failures
**Improvement**: Add exponential backoff retry for:
- WandB API operations
- Dispatcher job submissions
- Network operations

### 5.2 Resource Management
**Issue**: Basic capacity management doesn't consider resource heterogeneity
**Improvement**: 
- Track GPU memory requirements
- Support heterogeneous resource pools
- Dynamic resource allocation based on job requirements

### 5.3 Scheduler Sophistication
**Issue**: OptimizingScheduler waits for all incomplete jobs before scheduling new ones
**Improvement**:
- Implement ASHA (Asynchronous Successive Halving)
- Support Population-Based Training (PBT)
- Add early stopping mechanisms
- Enable parallel evaluation scheduling

### 5.4 Store Abstraction
**Issue**: Tightly coupled to WandB
**Improvement**:
- Add local SQLite store option
- Support for MLflow backend
- Abstract storage interface for cloud object stores
- Implement caching layer for frequently accessed data

### 5.5 Fault Recovery
**Issue**: Limited recovery from partial failures
**Improvement**:
- Checkpoint controller state periodically
- Implement job restart capabilities
- Add dead job detection and cleanup
- Support for resuming interrupted sweeps with full state

### 5.6 Optimization Features
**Issue**: Limited constraint handling and multi-objective support
**Improvement**:
- Add constraint satisfaction for invalid hyperparameter combinations
- Full multi-objective optimization with preference learning
- Implement transfer learning from previous sweeps
- Add meta-learning capabilities

### 5.7 Monitoring and Debugging
**Issue**: Debugging failed runs is challenging
**Improvement**:
- Add distributed tracing for job execution
- Implement log aggregation from distributed jobs
- Create debugging dashboard for sweep analysis
- Add performance profiling for optimization algorithms

### 5.8 Configuration Validation
**Issue**: Runtime failures from configuration errors
**Improvement**:
- Pre-flight validation of all configurations
- Dry-run mode for testing configurations
- Schema validation for recipe compatibility
- Automatic configuration migration tools

### 5.9 Scalability
**Issue**: Single controller bottleneck
**Improvement**:
- Distributed controller with leader election
- Sharded scheduling for large sweeps
- Batch operations for Store interactions
- Implement job queue with worker pools

### 5.10 Security and Isolation
**Issue**: Limited security considerations for cloud execution
**Improvement**:
- Secrets management for API keys
- Network isolation for experiments
- Resource quotas and limits
- Audit logging for compliance

---

## 6. Failure Scenario Analysis

### 6.1 Training Run Failure

**Current Behavior**:
1. Run state becomes FAILED in WandB
2. RunInfo.has_failed = True
3. JobStatus becomes FAILED
4. Scheduler sees failed status
5. Optimizer doesn't receive observation
6. Sweep continues with remaining trials

**Gaps**:
- No automatic retry of failed training
- No root cause analysis
- Failed runs count against max_trials

**Recommendations**:
- Implement configurable retry policy
- Distinguish transient vs permanent failures
- Add failure reason tracking

### 6.2 Evaluation Failure

**Current Behavior**:
1. Evaluation job fails (process crash or error)
2. Run remains in IN_EVAL or TRAINING_DONE_NO_EVAL
3. No observation recorded
4. Scheduler may try to schedule evaluation again (if not in dispatched_evals)

**Gaps**:
- No timeout for evaluation jobs
- No health checking during evaluation
- Missing evaluation doesn't fail the run

**Recommendations**:
- Add evaluation timeout with automatic failure
- Implement evaluation health checks
- Mark run as FAILED if evaluation fails repeatedly

### 6.3 Controller Failure and Reboot

**Current Behavior**:
1. Controller crashes (loses in-memory state)
2. On restart:
   - dispatched_trainings, dispatched_evals, completed_runs are empty
   - Fetches all runs from WandB
   - May re-dispatch jobs that are already running
   - Duplicate job submissions possible

**Critical Issues**:
- **Loss of dispatch tracking** leads to duplicate jobs
- **No detection of orphaned jobs**
- **Capacity management reset**

**Recommendations**:
- Persist controller state periodically
- Implement job deduplication at dispatcher level
- Add orphaned job detection and cleanup
- Store dispatch IDs in WandB for recovery

---

## 7. Retry Requirements

### 7.1 Critical Sections Requiring Retries

#### Store Operations (HIGH PRIORITY)
```python
# Current: No retries
self.store.init_run(job.run_id, sweep_id=self.sweep_id)
self.store.update_run_summary(job.run_id, {"suggestion": job.config})

# Needed: Exponential backoff with 3-5 attempts
```

#### WandB API Calls (HIGH PRIORITY)
```python
# fetch_runs - Network failures common
# update_run_summary - API rate limits possible
# init_run - Concurrent access issues
```

#### Dispatcher Operations (MEDIUM PRIORITY)
```python
# dispatch() - Resource allocation may fail transiently
# Skypilot launches - Cloud API failures
# Local subprocess creation - Resource exhaustion
```

### 7.2 Sections Needing Timeout Protection

#### Evaluation Jobs
```python
# No timeout currently - evaluations can hang indefinitely
# Need: Configurable timeout (default 1 hour)
# Action: Mark as FAILED, optionally retry
```

#### GP Model Training
```python
# Protein optimization can fail with numerical issues
# Need: Timeout and fallback to random sampling
# Current: Some fallback exists but could be improved
```

#### Control Loop Iterations
```python
# Individual iterations should have timeout
# Prevent single bad run from blocking sweep
# Need: Per-iteration timeout (default 5 minutes)
```

### 7.3 Recommended Retry Strategy

```python
class RetryConfig:
    max_attempts: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retry_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        HTTPError,
        WandbError,
    )

def with_retry(func, config: RetryConfig):
    """Decorator for retry logic with exponential backoff"""
    for attempt in range(config.max_attempts):
        try:
            return func()
        except config.retry_exceptions as e:
            if attempt == config.max_attempts - 1:
                raise
            delay = min(
                config.initial_delay * (config.backoff_factor ** attempt),
                config.max_delay
            )
            logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s")
            time.sleep(delay)
```

---

## Conclusions

The Metta sweep infrastructure demonstrates sophisticated design with clean architecture and robust state management. The protocol-based approach and separation of concerns provide excellent extensibility. The recent refactoring to use in-memory dispatch tracking elegantly solves WandB API caching issues.

Key strengths include the flexible execution modes, advanced optimization support, and comprehensive monitoring. However, there are opportunities for improvement in retry mechanisms, resource management, scheduler sophistication, and fault recovery.

Priority improvements should focus on:
1. **Retry mechanisms** for Store and Dispatcher operations
2. **Controller state persistence** for crash recovery
3. **Evaluation timeouts** to prevent hanging
4. **Advanced schedulers** (ASHA, PBT) for better efficiency
5. **Resource-aware scheduling** for heterogeneous clusters

The infrastructure provides a solid foundation for hyperparameter optimization experiments while maintaining room for growth in scalability, reliability, and advanced optimization features.