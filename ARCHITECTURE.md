# Metta AI - Comprehensive Architectural Map

## Table of Contents
1. [Overview](#overview)
2. [Core Components & Their Responsibilities](#core-components--their-responsibilities)
3. [Recipe System](#recipe-system)
4. [Data Flow](#data-flow)
5. [Supporting Infrastructure](#supporting-infrastructure)
6. [Key Entry Points](#key-entry-points)
7. [Development Workflows](#development-workflows)

---

## Overview

Metta AI is a reinforcement learning research project focusing on the emergence of cooperation and alignment in multi-agent AI systems. The architecture consists of:

- **Core simulation and RL infrastructure** (`metta/` directory)
- **C++/Python environment bindings** (`packages/mettagrid/`)
- **Recipe-based experiment system** (`recipes/`)
- **Training and evaluation tools** (`metta/tools/`)
- **Web dashboards and visualization** (`observatory/`, `gridworks/`)

### High-Level Architecture Flow

```
Recipes (Configuration)
    ↓
Tools (TrainTool, EvaluateTool, etc.)
    ↓
RL Core (Trainer, CheckpointManager, etc.)
    ↓
MettaGrid Environment (C++ Simulation + Python Bindings)
    ↓
Agent Policies (Neural Networks)
```

---

## Core Components & Their Responsibilities

### 1. **Main Entry Point: tools/run.py**

**Location**: `/Users/mp/Metta-ai/metta/tools/run.py`

The entry point for all user-facing operations. Routes invocations to recipe functions that return Tool instances.

```bash
./tools/run.py <tool> <recipe> [key=value ...]
./tools/run.py train arena run=my_experiment
```

- Uses `metta.common.tool.run_tool` for recipe discovery and tool loading
- Supports two-token form: `train arena` → `arena.train`
- Automatic argument classification (function args vs config overrides)
- Supports `--list` for discovery and `--dry-run` for validation

**Key Implementation**: `/Users/mp/Metta-ai/metta/metta/common/tool/run_tool.py`

---

### 2. **Tool Classes - Core Abstractions**

Tools are Pydantic config classes that inherit from `Tool` base class. Located in `/Users/mp/Metta-ai/metta/metta/tools/`.

#### **TrainTool** (`train.py`)
- **Purpose**: Configure and execute training runs
- **Key Responsibilities**:
  - Policy initialization and loading
  - Training environment setup (vectorization, parallelization)
  - Trainer component coordination (checkpointing, evaluation, profiling)
  - Distributed training orchestration
  - Weights & Biases integration
  
- **Config Fields**:
  - `run`: Run name (embedded in checkpoint URIs)
  - `trainer`: TrainerConfig (timesteps, batch size, learning rate, etc.)
  - `training_env`: Environment configuration (agents, seeds, curriculum)
  - `policy_architecture`: Neural network architecture
  - `evaluator`: Evaluation config with simulations
  - `checkpointer`: Policy checkpoint settings
  - `wandb`: Logging configuration

- **Key Methods**:
  - `invoke()`: Main entry point
  - `_load_or_create_policy()`: Initializes policy from scratch or loads from checkpoint
  - `_initialize_trainer()`: Creates Trainer with all components

#### **EvaluateTool** (`eval.py`)
- **Purpose**: Evaluate trained policies on simulation suites
- **Key Responsibilities**:
  - Policy loading from URIs (file://, s3://, wandb://)
  - Running simulations with different configurations
  - Collecting statistics (rewards, episode metrics)
  - Generating replay logs for visualization
  - Logging results to WandB
  
- **Config Fields**:
  - `simulations`: List of SimulationConfig to run
  - `policy_uris`: Policies to evaluate
  - `replay_dir`: Where to save replay logs
  - `stats_db_uri`: Where to export statistics

#### **PlayTool** (`play.py`)
- Interactive browser-based tool for manual testing
- Uses MettaScope visualization

#### **ReplayTool** (`replay.py`)
- Generates replay videos from saved episode logs
- Used for post-hoc visualization and debugging

#### **AnalysisTool** (`analyze.py`)
- Analyzes evaluation results
- Generates metrics and scorecard summaries

---

### 3. **Trainer - Core Training Loop**

**Location**: `/Users/mp/Metta-ai/metta/metta/rl/trainer.py`

The main coordinator for all training operations. Implements the core training loop with pluggable components.

#### **Architecture**:

```
Trainer (Main Facade)
    ├── Policy (Neural Network)
    ├── TrainingEnvironment (Vectorized Agents)
    ├── Experience Buffer (Replay Memory)
    ├── Optimizer (PyTorch)
    ├── CoreTrainingLoop (Forward/Backward Pass)
    ├── Components (Pluggable):
    │   ├── Checkpointer (Policy snapshots)
    │   ├── ContextCheckpointer (Full training state)
    │   ├── Evaluator (Run policies on simulations)
    │   ├── StatsReporter (Log metrics to wandb/server)
    │   ├── Scheduler (Learning rate scheduling)
    │   ├── Uploader (Policy upload to S3)
    │   ├── TorchProfiler (Performance profiling)
    │   └── WandbLogger (Training run tracking)
    └── ComponentContext (Shared state across all components)
```

#### **Key Concepts**:

- **TrainerComponent**: Interface for plugins that hook into training
- **ComponentContext**: Shared mutable state (policy, optimizer, losses, etc.) accessible to all components
- **TrainerState**: Tracks timesteps, epochs, and training progress
- **CoreTrainingLoop**: Runs forward pass, computes losses, backprop, optimization step

#### **Training Flow**:
1. Initialize policy and components
2. Run vectorized environment to collect experience
3. Compute losses (PPO, contrastive, etc.)
4. Optimization step
5. Checkpointer saves policies and trainer state periodically
6. Evaluator runs policies on simulations if configured
7. Repeat until total_timesteps reached

**Key Classes**:
- `Trainer`: Main facade coordinating training
- `CoreTrainingLoop`: Inner loop with forward/backward
- `TrainerState`: Training progress tracking
- `ComponentContext`: Shared component state

---

### 4. **CheckpointManager - Policy Persistence**

**Location**: `/Users/mp/Metta-ai/metta/metta/rl/checkpoint_manager.py`

Manages loading and saving of policies with URI-based addressing.

#### **Key Responsibilities**:
- Load policies from multiple sources (file, S3, mock)
- Save policies with versioning (epoch numbers)
- Handle metadata extraction from URIs
- Support remote S3 storage with local caching

#### **URI Format**:
```
file://./train_dir/run_name/checkpoints/run_name:v5.mpt
s3://bucket/path/run_name/checkpoints/run_name:v10.mpt
mock://test_agent
```

- Filename embeds run name and epoch
- Enables version tracking without database
- Latest checkpoint discoverable via `_latest_checkpoint()`

#### **Key Methods**:
- `load_from_uri(uri, policy_env_info, device)`: Load policy artifact
- `save_checkpoint()`: Save with versioning
- `load_artifact_from_uri()`: Load PolicyArtifact (policy + metadata)
- `normalize_uri()`: Resolve to canonical form

---

### 5. **Policies - Neural Network Implementations**

**Location**: `/Users/mp/Metta-ai/metta/agent/src/metta/agent/`

Agents are PyTorch neural networks implementing the `Policy` interface.

#### **Policy Hierarchy**:

```
Policy (Abstract Base)
├── Implements:
│   ├── TrainablePolicy (mettagrid interface)
│   ├── nn.Module (PyTorch)
├── Abstract Methods:
│   ├── forward(td, action) -> TensorDict
│   ├── reset_memory()
│   ├── get_agent_experience_spec()
├── Implementations:
│   ├── Fast (CNN-based, token-to-box)
│   ├── ViT variants (Attention-based)
│   │   ├── ViTDefaultConfig
│   │   ├── ViTSmallConfig
│   │   ├── ViTMediumConfig
│   │   └── ViTTinyConfig
│   ├── LSTM-based architectures
│   ├── Cortex (Complex multi-component)
│   └── Mamba-based (State-space models)
```

#### **Key Concepts**:

- **PolicyArchitecture**: Configuration class specifying network components
- **PolicyEnvInterface**: Adapter providing observation/action space info
- **Components**: Reusable layers (attention, convolution, token encoding)
- **TensorDict**: Data structure passed through policy (observations, actions, values)

#### **Policy Types**:
- **Fast**: CNN-based, highest throughput, lower expressivity
- **ViT (Vision Transformer)**: Attention-based, adaptable to changing environments, three size variants
- **LSTM**: Memory-enabled, slower but good for partial observability
- **Mamba**: State-space models, balance of speed and expressivity

#### **Files**:
- `/Users/mp/Metta-ai/metta/agent/src/metta/agent/policy.py`: Base Policy class
- `/Users/mp/Metta-ai/metta/agent/src/metta/agent/policies/`: Policy implementations
- `/Users/mp/Metta-ai/metta/agent/src/metta/agent/components/`: Reusable neural components

---

### 6. **Training Environment - Experience Collection**

**Location**: `/Users/mp/Metta-ai/metta/metta/rl/training/training_environment.py`

Wraps the MettaGrid environment for training with vectorization and batch processing.

#### **Responsibilities**:
- Vectorize multiple environment instances
- Manage agent populations across parallel environments
- Handle curriculum (dynamic environment difficulty)
- Collect raw experience (observations, actions, rewards)
- Manage RNN hidden states across timesteps

#### **Key Classes**:
- `VectorizedTrainingEnvironment`: Wraps PufferLib vectorization
- `TrainingEnvironmentConfig`: Configuration
- `Curriculum`: Optional progressive difficulty system

#### **Experience Flow**:
```
Training Env (N parallel instances)
    ↓
Experience Buffer (batch_size, sequence_length)
    ↓
Losses compute gradients
    ↓
Optimizer.step()
```

---

### 7. **Simulation - Evaluation Infrastructure**

**Location**: `/Users/mp/Metta-ai/metta/metta/sim/simulation.py`

Runs policies in controlled evaluation scenarios (independent of training).

#### **Key Responsibilities**:
- Load policies from URIs
- Run episodes with fixed configurations
- Collect statistics (rewards, episode length, win rate)
- Generate replay logs for visualization
- Support multiple policy types (learned, random, mock)

#### **Architecture**:
```
Simulation (Config + Policy)
    ├── Load PolicyArtifact from URI
    ├── Create environment instance
    ├── For each episode:
    │   ├── Reset environment
    │   ├── Run rollout with policy
    │   ├── Collect episode stats
    │   └── Write replay log
    └── Export to stats database
```

#### **Key Methods**:
- `simulate()`: Run configured number of episodes
- `_run_episode()`: Single episode execution
- Statistics collection → DuckDB

#### **SimulationConfig**:
- `suite`: Category (arena, navigation, etc.)
- `name`: Specific scenario
- `env`: MettaGridConfig with environment parameters
- `num_episodes`: Episodes to run
- `max_time_s`: Wall-clock timeout

---

### 8. **Loss Functions - Training Objectives**

**Location**: `/Users/mp/Metta-ai/metta/metta/rl/loss/`

Pluggable loss modules computing gradients from experience.

#### **Available Losses**:
- **PPO** (`ppo.py`): Policy gradient with value function baseline
- **GRPO** (`grpo.py`): Group reward policy optimization
- **Contrastive** (`contrastive.py`): Self-supervised representation learning
- **Action Supervised** (`action_supervised.py`): Imitation learning from demonstrations
- **Dynamics** (`dynamics.py`): World model learning
- **SL Kickstarter** (`sl_kickstarter.py`): Supervised learning initialization

#### **Loss Interface**:
- Processes Experience (observations, actions, rewards, values)
- Returns scalar loss for backpropagation
- Optionally logs metrics
- Can modify TensorDict between layers

---

### 9. **MettaGrid Environment - Simulation Core**

**Location**: `/Users/mp/Metta-ai/metta/packages/mettagrid/`

C++ environment with Python bindings. Provides high-performance multi-agent gridworld simulation.

#### **Architecture**:

```
C++ Core (fast simulation)
    ├── Game mechanics
    ├── Physics/collisions
    ├── Agent state management
    └── Reward computation

Pybind11 Bindings
    ↓
Python API (Gym-like)
    ├── PufferMettaGridEnv (Primary - for training)
    ├── MettaGridGymEnv (Gymnasium compatibility)
    ├── MettaGridPettingZooEnv (PettingZoo compatibility)
    └── MettaGridPufferEnv (PufferLib compatibility)
```

#### **Key Features**:
- **Vectorized Rollout**: Efficient multi-episode execution
- **Replay Recording**: Built-in episode logging (JSON format)
- **Curriculum Support**: Dynamic environment configuration changes
- **Configurable**: YAML-based game rules

#### **Objects**:
- Agents (controllable entities with energy/resources)
- Altar (reward generation via energy exchange)
- Generator (resource harvesting)
- Converter (resource → energy conversion)
- Wall, Shield, Attack mechanics

#### **Key Python Files**:
- `/Users/mp/Metta-ai/metta/packages/mettagrid/python/src/mettagrid/policy/policy.py`: Policy interface
- `/Users/mp/Metta-ai/metta/packages/mettagrid/python/src/mettagrid/simulator/`: Simulation runner
- `/Users/mp/Metta-ai/metta/packages/mettagrid/python/src/mettagrid/builder/envs.py`: Environment builders

---

### 10. **Statistics & Analysis**

**Location**: `/Users/mp/Metta-ai/metta/metta/sim/stats/` and `/Users/mp/Metta-ai/metta/metta/eval/`

Post-run analysis and metric computation.

#### **Components**:

- **SimulationStatsDB**: DuckDB-based storage for episode statistics
  - Episode rewards, lengths, agent actions
  - Mergeable across multiple simulations
  - Fast querying and aggregation

- **EvalStatsDB**: Evaluation-level aggregations
  - Policy-level scores
  - Win rates, average rewards per scenario

- **Replay Recording** (`ReplayLogWriter`):
  - JSON format recordings of episodes
  - Used by MettaScope visualization
  - Contains frame-by-frame observations/actions

---

## Recipe System

### Overview

Recipes are Python modules in `recipes/` that define "tool makers" - functions returning configured Tool instances.

**Purpose**: Bundle related configurations (train, eval, play) ensuring consistency across the workflow.

### Structure

```
recipes/
├── prod/                      # Production-ready, validated
│   ├── arena_basic_easy_shaped.py
│   └── cvc/
│       └── small_maps.py
├── experiment/                # Work-in-progress
│   ├── abes/                  # Architecture experiments
│   ├── losses/                # Loss function experiments
│   └── ...
├── validation/                # CI/stable test definitions
│   ├── ci_suite.py           # Lightweight smoke tests
│   └── stable_suite.py        # Performance validation
└── common/                    # Shared utilities
```

### Anatomy of a Recipe

```python
# recipes/prod/arena_basic_easy_shaped.py

from metta.tools.train import TrainTool
from metta.tools.eval import EvaluateTool
from metta.sim.simulation_config import SimulationConfig

def mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Shared environment config across all tools."""
    return eb.make_arena(num_agents=num_agents)

def simulations() -> list[SimulationConfig]:
    """Shared simulations for eval and analysis."""
    env = mettagrid()
    return [
        SimulationConfig(suite="arena", name="basic", env=env),
        SimulationConfig(suite="arena", name="combat", env=env.copy()),
    ]

def train() -> TrainTool:
    """Training tool with curriculum and evaluation."""
    return TrainTool(
        training_env=TrainingEnvironmentConfig(...),
        evaluator=EvaluatorConfig(simulations=simulations()),
    )

def evaluate() -> EvaluateTool:
    """Evaluation tool (same simulations as training)."""
    return EvaluateTool(simulations=simulations())

def play() -> PlayTool:
    """Interactive play (first simulation)."""
    return PlayTool(sim=simulations()[0])
```

### Prod vs Experiment Recipes

#### **Prod Recipes** (`recipes/prod/`)
- **Stability**: Validated to work consistently
- **CI Tests**: Included in `ci_suite.py` for smoke testing
- **Performance Baselines**: Tracked in `stable_suite.py`
- **Examples**: `arena_basic_easy_shaped`, `cvc.small_maps`

#### **Experiment Recipes** (`recipes/experiment/`)
- **Work-in-progress**: No stability guarantees
- **Exploration**: New architectures, loss functions, curriculum ideas
- **Import flexibility**: Can depend on prod recipes
- **Categories**:
  - `abes/`: Architecture experiments
  - `losses/`: Loss function variations
  - `scratchpad/`: User-specific experiments (git-ignored)

### Validation System

#### **CI Suite** (`recipes/validation/ci_suite.py`)
- **Purpose**: Quick smoke tests (every commit)
- **Job Type**: `JobConfig` instances
- **Characteristics**:
  - Low timesteps (10k for training)
  - Short timeouts (5 minutes)
  - Goal: Verify no crashes, basic functionality
  - Run on GitHub before merge (planned)

**Example CI Job**:
```python
arena_train = JobConfig(
    name="ci.arena_train",
    module="recipes.prod.arena_basic_easy_shaped.train",
    args=[
        "run=ci.arena_train",
        "trainer.total_timesteps=10000",  # Quick test
    ],
    timeout_s=300,  # 5 minutes
    is_training_job=True,
)
```

#### **Stable Suite** (`recipes/validation/stable_suite.py`)
- **Purpose**: Performance validation during releases
- **Characteristics**:
  - Full training runs (100M-2B timesteps)
  - Remote multi-GPU infrastructure
  - Tracks SPS, learning curves, final performance
  - Acceptance criteria (e.g., SPS >= 40k steps/sec)

**Example Stable Job**:
```python
arena_train_100m = JobConfig(
    name="stable.arena_train_100m",
    module="recipes.prod.arena_basic_easy_shaped.train",
    args=["trainer.total_timesteps=100000000"],
    timeout_s=7200,
    remote=RemoteConfig(gpus=1),
    metrics_to_track=["overview/sps", "env_agent/heart.gained"],
    acceptance_criteria=[
        AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
    ],
)
```

### Recipe Discovery

The tool runner uses `pkgutil.walk_packages()` to auto-discover recipes:
1. Scan `recipes/` subdirectories
2. Import modules matching `*_config.py` or tool-makers
3. Build registry mapping recipe names to modules
4. Load tool-maker functions on demand

**Important**: Recipe subdirectories need `__init__.py` files:
```bash
uv run python devops/tools/ensure_recipe_packages.py
```

---

## Data Flow

### Training Pipeline

```
1. Recipe Definition
   └─> TrainTool configuration (learning rate, batch size, etc.)

2. Policy Initialization
   ├─> Load from checkpoint URI (if initial_policy_uri set)
   └─> Create new policy from architecture config

3. Environment Setup
   ├─> VectorizedTrainingEnvironment wraps MettaGrid
   ├─> Parallelization via PufferLib
   └─> Initialize curriculum (if enabled)

4. Training Loop (Trainer.train())
   ├─> Sample experience (vectorized environment)
   │   └─> Collect observations, actions, rewards, dones
   ├─> Compute losses (PPO, contrastive, etc.)
   ├─> Backward pass & optimization
   ├─> Trainer components execute (checkpointer, evaluator, etc.)
   └─> Repeat until total_timesteps reached

5. Policy Checkpointing
   ├─> CheckpointManager saves to local dir
   ├─> Filename: <run_name>:v<epoch>.mpt
   └─> Optional S3 upload (if remote_prefix configured)

6. Periodic Evaluation (if evaluator.epoch_interval > 0)
   ├─> Evaluator loads latest checkpoint
   ├─> Runs simulations for each evaluation scenario
   ├─> Collects statistics to stats_db
   └─> Logs metrics to WandB

7. Training Completion
   └─> Final policy saved
       └─> Accessible via file:// or s3:// URI
```

### Evaluation Pipeline

```
1. Policy Loading
   ├─> Parse URI (file://, s3://, wandb://)
   ├─> CheckpointManager.load_from_uri()
   └─> Extract run_name and epoch from filename

2. Simulation Execution (per simulation config)
   ├─> Initialize environment instance
   ├─> For each episode:
   │   ├─> Reset environment & policy memory
   │   ├─> Rollout with policy
   │   ├─> Collect step-by-step stats
   │   └─> Record replay (if enabled)
   └─> Write to stats database

3. Statistics Collection
   ├─> Episode rewards
   ├─> Episode lengths
   ├─> Agent action frequencies
   └─> Stored in DuckDB

4. Merge Simulations
   └─> Combine results from all scenarios into one stats_db

5. Results Export
   ├─> Generate scorecard (win rates, avg rewards)
   ├─> Log to WandB (if enabled)
   ├─> Export stats to S3 or local path
   └─> Return EvalResults object
```

### Component Communication

#### **Policy URIs**
Central abstraction for policy addressing:
```
Filename:     <run_name>:v<epoch>.mpt
File URI:     file://./train_dir/run_name/checkpoints/run_name:v5.mpt
S3 URI:       s3://bucket/path/run_name/checkpoints/run_name:v10.mpt
WandB URI:    wandb://run/my_run/artifacts/run_name:v3.mpt
```

Enables:
- Version tracking without database
- Easy URI composition
- Metadata extraction from filename
- S3 caching at local path

#### **ComponentContext**
Shared mutable state accessible to all training components:
```
ComponentContext:
├── policy: Current trainable policy
├── optimizer: PyTorch optimizer
├── config: TrainerConfig
├── experience: Experience buffer
├── losses: Loss functions
├── state: TrainerState (timesteps, epochs)
└── ... (other components read/write as needed)
```

---

## Supporting Infrastructure

### 1. **Visualization Tools**

#### **MettaScope** (Interactive Replay Viewer)
- **Location**: `/Users/mp/Metta-ai/metta/packages/mettagrid/nim/mettascope/`
- **Technology**: Nim-based UI (compiled to WebAssembly)
- **Purpose**: Frame-by-frame visualization of episodes
- **Input**: JSON replay logs from `ReplayLogWriter`
- **Features**:
  - Play/pause/step through episodes
  - Agent-centric views
  - State inspection at each step
  - Interactive demo with live policies

#### **Observatory** (Training Dashboard)
- **Location**: `/Users/mp/Metta-ai/metta/observatory/`
- **Technology**: React-based web dashboard
- **Purpose**: Monitor training runs in real-time
- **Features**:
  - WandB integration for metrics
  - Training curves (SPS, rewards, losses)
  - Policy comparison
  - Evaluation result browsing

#### **GridWorks** (Web Interface)
- **Location**: `/Users/mp/Metta-ai/metta/gridworks/`
- **Technology**: Next.js (React)
- **Purpose**: General-purpose web interface for experiment management
- **Features**:
  - Run configuration & submission
  - Results browsing
  - Policy selection for evaluation

### 2. **Configuration System**

The project uses **OmegaConf** for hierarchical configuration management.

#### **Configuration Hierarchy**:
```
Base Configs (env defaults)
    ↓
Recipe Tool Instances
    ↓
CLI Argument Overrides
    ↓
Final Configuration
```

#### **Key Config Areas**:

- **Agent Architecture** (in policies):
  - Network size/complexity
  - Component choices (attention, CNN, LSTM, etc.)
  - Example: `ViTDefaultConfig` in policy

- **Trainer Config** (`TrainerConfig`):
  - Learning rate, batch size, optimizer type
  - Loss function weights
  - Gradient clipping, warmup schedule
  - Total timesteps

- **Training Environment** (`TrainingEnvironmentConfig`):
  - Number of parallel environments
  - Agent count, seed
  - Curriculum configuration

- **Evaluation Config** (`EvaluatorConfig`):
  - SimulationConfig list
  - Epoch interval for periodic evaluation
  - Local vs remote evaluation choice

### 3. **Analysis Infrastructure**

#### **Statistics Database** (DuckDB)
- Episode-level statistics (rewards, lengths)
- Fast SQL querying and aggregation
- Supports merging across simulations
- Schema:
  ```sql
  episodes (id, policy_uri, simulation, episode_idx, reward, length, ...)
  steps (id, episode_id, agent_id, action, observation, ...)
  ```

#### **Analysis Tools** (`metta/eval/analysis.py`)
- Generates scorecard (win rates, mean rewards by scenario)
- Computes confidence intervals
- Exports to JSON/CSV

#### **Marimo Notebooks** (`analysis/marimo/`)
- Interactive Python notebooks
- Real-time statistics from stats_db
- Visualization with Plotly

#### **Jupyter Notebooks** (`notebooks/`)
- Traditional notebooks for experimentation
- Analysis utilities in `analysis/analysis.py`

### 4. **Distributed Training**

#### **DistributedHelper** (`metta/rl/training/distributed_helper.py`)
- Abstracts torch.distributed setup
- Multi-GPU/Multi-node training
- Gradient synchronization
- Checkpoint aggregation

#### **Key Features**:
- Automatic batch scaling across GPUs
- Distributed data parallelism
- Gradient averaging
- Rank-aware logging

### 5. **Infrastructure Components**

#### **Checkpointing**
- **Checkpointer**: Saves policy at regular intervals
- **ContextCheckpointer**: Saves full training state (optimizer, scheduler, curriculum)
- Automatic state serialization with SafeTensors format

#### **Monitoring & Logging**
- **StatsReporter**: Sends metrics to WandB, HTTP server
- **ProgressLogger**: Console output with training progress
- **WandbLogger**: Integration with Weights & Biases
- **Gradient Reporter**: Monitors gradient norms

#### **Performance**
- **TorchProfiler**: PyTorch profiler integration
- **Stopwatch**: Timing instrumentation
- **FLOPS tracking**: Throughput monitoring

#### **Remote Infrastructure**
- **Uploader**: Pushes checkpoints to S3
- **stats_client**: HTTP client for stats server
- **Remote evaluation**: Dispatch policies to remote workers

---

## Key Entry Points

### 1. **User-Facing: tools/run.py**

```bash
./tools/run.py <tool> <recipe> [args...]
./tools/run.py train arena run=my_run trainer.total_timesteps=1000000
./tools/run.py evaluate arena policy_uris=file://./train_dir/my_run/checkpoints
./tools/run.py play arena
./tools/run.py analyze arena
```

**Flow**:
1. Parse command line
2. Load recipe module (auto-discovery or explicit path)
3. Find tool-maker function (tool_name)
4. Call tool-maker to get Tool instance
5. Apply CLI argument overrides (OmegaConf merge)
6. Call `tool.invoke()` to execute

### 2. **Programmatic: Recipe Functions**

```python
# In recipes/prod/arena_basic_easy_shaped.py
from metta.tools.train import TrainTool

def train() -> TrainTool:
    return TrainTool(
        training_env=TrainingEnvironmentConfig(...),
        policy_architecture=ViTDefaultConfig(),
        evaluator=EvaluatorConfig(simulations=simulations()),
    )
```

Direct instantiation allows:
- Embedding in notebooks
- Integration with other tools
- Batch job submission
- Programmatic configuration

### 3. **Recipe Job Configuration**

```python
# In recipes/validation/ci_suite.py
from metta.jobs.job_config import JobConfig

def get_ci_jobs() -> tuple[list[JobConfig], str]:
    return [
        JobConfig(
            name="ci.arena_train",
            module="recipes.prod.arena_basic_easy_shaped.train",
            args=["trainer.total_timesteps=10000"],
            timeout_s=300,
        ),
    ]
```

Job runner executes via `tools/run.py` with module path and args.

### 4. **Training Direct**

```python
# Manual training loop (advanced)
from metta.rl.trainer import Trainer
from metta.rl.training import VectorizedTrainingEnvironment
from metta.agent.policy import Policy

env = VectorizedTrainingEnvironment(config)
policy = Policy.make(architecture, env)
trainer = Trainer(config, env, policy, device)
trainer.train()
```

---

## Development Workflows

### 1. **Training a Model**

```bash
# Quick test (1M steps)
export TEST_ID=$(date +%Y%m%d_%H%M%S)
./tools/run.py train arena run=test_$TEST_ID trainer.total_timesteps=1000000

# Prod run (100M steps)
./tools/run.py train arena_basic_easy_shaped run=prod_run_1 trainer.total_timesteps=100000000

# Custom configuration
./tools/run.py train arena \
  run=my_exp \
  trainer.total_timesteps=50000000 \
  trainer.batch_size=512 \
  policy_architecture=latent_attn_small \
  trainer.learning_rate=0.001
```

### 2. **Evaluating a Policy**

```bash
# Evaluate latest checkpoint from training run
./tools/run.py evaluate arena \
  policy_uris=file://./train_dir/test_$TEST_ID/checkpoints

# Evaluate specific version
./tools/run.py evaluate arena \
  policy_uris=file://./train_dir/my_run/checkpoints/my_run:v42.mpt

# Evaluate multiple policies
./tools/run.py evaluate arena \
  policy_uris='[
    "file://./train_dir/run1/checkpoints/run1:v10.mpt",
    "file://./train_dir/run2/checkpoints/run2:v20.mpt"
  ]'

# Evaluate S3 policy
./tools/run.py evaluate arena \
  policy_uris=s3://my-bucket/policies/run_name/checkpoints/run_name:v10.mpt
```

### 3. **Interactive Testing**

```bash
# Play with trained policy (browser-based)
./tools/run.py play arena policy_uri=file://./train_dir/my_run/checkpoints

# Replay recorded episode
./tools/run.py replay arena policy_uri=file://./train_dir/my_run/checkpoints
```

### 4. **Creating a New Recipe**

**Steps**:
1. Create `recipes/experiment/my_recipe.py` or `recipes/prod/my_recipe.py`
2. Define helper functions:
   - `mettagrid()` - shared environment config
   - `simulations()` - evaluation scenarios
3. Define tool makers:
   - `train() -> TrainTool`
   - `evaluate() -> EvaluateTool`
   - `play() -> PlayTool`
   - `replay() -> ReplayTool`
4. Run:
   ```bash
   ./tools/run.py train my_recipe run=test
   ./tools/run.py --list  # Discover available recipes
   ```

### 5. **CI/Validation Testing**

```bash
# Run all CI tests locally
metta ci --stage recipe-tests

# Run just smoke tests
uv run python devops/tools/ensure_recipe_packages.py  # Setup __init__.py files first

# Validate a specific recipe
./tools/run.py recipes.prod.arena_basic_easy_shaped.train \
  run=ci_test \
  trainer.total_timesteps=10000 \
  trainer.batch_size=32
```

### 6. **Code Quality**

```bash
# Full CI pipeline
metta ci

# Python tests only
metta pytest

# Lint and auto-fix
metta lint --fix metta/

# Format staged files
metta lint --staged --fix
```

---

## Architecture Decisions & Trade-offs

### 1. **Recipe System**
- **Why**: Bundle related configs, ensure consistency
- **Trade-off**: Extra indirection vs single configuration file
- **Benefit**: Easy to maintain multiple variants (prod, experiment)

### 2. **URI-Based Policy Addressing**
- **Why**: Version tracking without database
- **Trade-off**: Metadata in filename vs structured storage
- **Benefit**: Simple, portable, self-contained

### 3. **Pluggable Trainer Components**
- **Why**: Extensibility without core changes
- **Trade-off**: More code vs monolithic trainer
- **Benefit**: Easy to add checkpointing, profiling, monitoring

### 4. **DuckDB for Statistics**
- **Why**: Fast local queries, no server setup
- **Trade-off**: Local-only vs centralized tracking
- **Benefit**: Portable, mergeable, SQL-queryable

### 5. **Separate Training & Evaluation Environments**
- **Why**: Clean separation of concerns
- **Trade-off**: Policy loading overhead vs code reuse
- **Benefit**: Independent scaling, different optimization goals

---

## File Organization Summary

```
/Users/mp/Metta-ai/metta/
├── metta/                           # Core Python package
│   ├── tools/                       # Tool implementations
│   │   ├── train.py                 # Training tool
│   │   ├── eval.py                  # Evaluation tool
│   │   ├── play.py                  # Interactive play
│   │   ├── replay.py                # Replay visualization
│   │   ├── analyze.py               # Analysis tool
│   │   └── utils/                   # Utilities
│   ├── rl/                          # Reinforcement learning core
│   │   ├── trainer.py               # Main training loop
│   │   ├── trainer_config.py        # Training configuration
│   │   ├── checkpoint_manager.py    # Policy persistence
│   │   ├── training/                # Training components
│   │   │   ├── core.py              # Core training loop
│   │   │   ├── checkpointer.py      # Checkpoint saving
│   │   │   ├── evaluator.py         # Policy evaluation
│   │   │   ├── stats_reporter.py    # Metrics logging
│   │   │   └── ...
│   │   └── loss/                    # Loss functions (PPO, etc.)
│   ├── sim/                         # Simulation & evaluation
│   │   ├── simulation.py            # Simulation runner
│   │   ├── simulation_config.py     # Simulation configuration
│   │   ├── stats/                   # Statistics collection
│   │   └── replay_log_writer.py     # Replay recording
│   ├── eval/                        # Evaluation infrastructure
│   │   ├── eval_service.py          # Evaluation orchestration
│   │   └── analysis.py              # Results analysis
│   ├── common/                      # Common utilities
│   │   ├── tool/                    # Tool runner
│   │   │   └── run_tool.py          # Main entry point logic
│   │   └── wandb/                   # WandB integration
│   ├── evals/                       # Remote evaluation service
│   └── setup/                       # Installation & configuration
├── agent/                           # Agent/policy package
│   └── src/metta/agent/
│       ├── policy.py                # Base Policy class
│       ├── policies/                # Policy implementations
│       │   ├── vit.py               # Vision Transformer policies
│       │   ├── fast.py              # Fast CNN policy
│       │   ├── cortex.py            # Complex multi-component
│       │   └── ...
│       └── components/              # Neural network components
├── packages/mettagrid/              # Environment bindings
│   ├── cpp/                         # C++ core (Bazel)
│   ├── python/                      # Python bindings
│   │   └── src/mettagrid/
│   │       ├── policy/              # Policy interfaces
│   │       ├── simulator/           # Simulation runner
│   │       └── builder/             # Environment builders
│   └── nim/mettascope/             # Replay viewer (Nim)
├── recipes/                         # Experiment configurations
│   ├── prod/                        # Production recipes
│   │   ├── arena_basic_easy_shaped.py
│   │   └── cvc/
│   ├── experiment/                  # Experimental recipes
│   │   ├── abes/
│   │   ├── losses/
│   │   └── ...
│   └── validation/                  # CI/stable test definitions
│       ├── ci_suite.py
│       └── stable_suite.py
├── observatory/                     # Training dashboard (React)
├── gridworks/                       # Web interface (Next.js)
├── app_backend/                     # FastAPI backend server
├── tools/                           # CLI tools
│   └── run.py                       # Main entry point (./tools/run.py)
├── tests/                           # Unit & integration tests
│   ├── rl/                          # RL component tests
│   ├── sim/                         # Simulation tests
│   └── ...
├── notebooks/                       # Jupyter notebooks
├── analysis/                        # Analysis tools
│   ├── marimo/                      # Interactive notebooks
│   └── analysis.py                  # Analysis utilities
├── CLAUDE.md                        # Project guidelines (THIS FILE)
└── README.md                        # Project overview
```

---

## Key Patterns & Best Practices

### 1. **Policy Loading Pattern**
```python
from metta.rl.checkpoint_manager import CheckpointManager
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

policy_env_info = PolicyEnvInterface.from_mg_cfg(env_config)
policy = CheckpointManager.load_from_uri(uri, policy_env_info, device)
```

### 2. **Tool Creation Pattern**
```python
from metta.tools.train import TrainTool

def my_train() -> TrainTool:
    return TrainTool(
        training_env=...,
        policy_architecture=...,
        evaluator=...,
    )
```

### 3. **Simulation Configuration Pattern**
```python
from metta.sim.simulation_config import SimulationConfig

sims = [
    SimulationConfig(
        suite="arena",
        name="scenario1",
        env=mettagrid_config,
        num_episodes=100,
    ),
]
```

### 4. **Recipe Structure Pattern**
```python
# Shared config
def mettagrid(): ...
def simulations(): ...

# Tool makers
def train() -> TrainTool: ...
def evaluate() -> EvaluateTool: ...
def play() -> PlayTool: ...
```

---

## Troubleshooting Guide

### **Missing Dependencies**
```bash
metta install              # Reinstall all components
metta install core         # Core dependencies only
```

### **Build Issues**
```bash
metta clean                # Clean build artifacts
metta configure            # Reconfigure for your system
```

### **Import Errors**
Ensure recipe packages have `__init__.py`:
```bash
uv run python devops/tools/ensure_recipe_packages.py
```

### **Policy Not Found**
Check URI format and ensure checkpoint file exists:
```bash
ls ./train_dir/my_run/checkpoints/
# Should see: my_run:v0.mpt, my_run:v1.mpt, etc.
```

---

## Resources

- **README**: `/Users/mp/Metta-ai/metta/README.md` - Project overview and quick start
- **CLAUDE.md**: `/Users/mp/Metta-ai/metta/CLAUDE.md` - Development guidelines (in codebase)
- **Tool Runner Docs**: `/Users/mp/Metta-ai/metta/common/src/metta/common/tool/README.md`
- **MettaGrid README**: `/Users/mp/Metta-ai/metta/packages/mettagrid/README.md` - Environment details
- **Agent README**: `/Users/mp/Metta-ai/metta/agent/README.md` - Policy architecture info
- **Roadmap**: `/Users/mp/Metta-ai/metta/roadmap.md` - Research directions
