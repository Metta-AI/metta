# Metta Tools Documentation

This document provides comprehensive documentation for all tools in the `./tools/` directory. These tools provide
essential functionality for training, evaluation, visualization, and development workflows in the Metta ecosystem.

## Quick Reference Table

| Category          | Tool                         | Purpose                                       | GPU Required | Database Access |
| ----------------- | ---------------------------- | --------------------------------------------- | ------------ | --------------- |
| **Training**      | `run.py train <recipe>`      | Train policies with recipe configurations     | ✓            | Optional        |
|                   | `sweep_init.py`                         | Initialize hyperparameter sweep experiments   | ✗            | ✗               |
|                   | `sweep_eval.py`                         | Evaluate policies from sweep runs             | ✓            | ✗               |
| **Evaluation**    | `run.py eval <recipe>`      | Run policy evaluation with recipe system      | ✓            | ✓               |
|                   | `run.py analyze <recipe>`   | Analyze evaluation results with recipes       | ✗            | ✓               |
| **Visualization** | `run.py play <recipe>`      | Interactive gameplay via recipe system        | ✗            | ✗               |
|                   | `run.py replay <recipe>`    | Generate replay files via recipe system       | ✓            | ✗               |
|                   | `renderer.py`                           | Real-time ASCII/Miniscope rendering (legacy)  | ✓            | ✗               |
|                   | `dashboard.py`                          | Generate dashboard data for web visualization | ✗            | ✓               |
| **Map Tools**     | `map/gen.py`                            | Generate maps from configuration files        | ✗            | ✗               |
|                   | `map/gen_scene.py`                      | Generate maps from scene templates            | ✗            | ✗               |
|                   | `map/view.py`                           | View stored maps in various formats           | ✗            | ✗               |
|                   | `map/normalize_ascii_map.py`            | Normalize ASCII map characters                | ✗            | ✗               |
|                   | `map/normalize_scene_patterns.py`       | Normalize WFC/ConvChain patterns              | ✗            | ✗               |
| **Utilities**     | `validate_config.py`                    | Validate and print Hydra configurations       | ✗            | ✗               |
|                   | `stats_duckdb_cli.py`                   | Interactive DuckDB CLI for stats analysis     | ✗            | ✓               |
|                   | `upload_map_imgs.py`                    | Upload map images to S3                       | ✗            | ✗               |
|                   | `dump_src.py`                           | Dump source files for LLM context             | ✗            | ✗               |
|                   | `autotune.py`                           | Auto-tune vectorization parameters            | ✗            | ✗               |

## Tool Execution

The main entry point is `./tools/run.py` which uses the recipe system for all major operations:

```bash
./tools/run.py train arena run=my_experiment # Training
./tools/run.py eval arena policy_uri=s3://my-bucket/checkpoints/my_experiment/my_experiment:v20.pt # Evaluation
./tools/run.py play arena policy_uri=s3://my-bucket/checkpoints/my_experiment/my_experiment:v20.pt # Interactive play
```

### Recipe Structure

Recipe modules define explicit tool functions that return Tool instances:

```python
def train() -> TrainTool:
    return TrainTool(...)

def eval() -> EvalTool:
    return EvalTool(simulations=[...])
```

Recipes can optionally define helper functions like `mettagrid()` or `simulations()` to avoid duplication when multiple tools need the same configuration.

Examples:

```bash
# Non-remote evaluation
./tools/run.py eval arena policy_uris=mock://policy

# Remote evaluation
./tools/run.py eval_remote arena policy_uri=s3://bucket/run:v10.pt
```

Shorthands:

- Omit prefix: `arena.train` == `experiments.recipes.arena.train`
- Two-token sugar: `train arena` == `arena.train`

Note: The evaluation tool class is `EvalTool` (renamed from `EvalTool`). If importing directly, use:

```python
from metta.tools.eval import EvalTool
```

## Training Tools

### train.py

**Purpose**: Train Metta policies using PPO (Proximal Policy Optimization) algorithm.

This is the primary training tool in the Metta ecosystem, supporting:

- Distributed training across multiple GPUs/nodes
- Automatic hyperparameter configuration via Hydra
- Real-time metrics tracking with Wandb
- Checkpoint saving and policy versioning
- Configurable evaluation during training

The tool integrates with:

- CheckpointManager: For saving and managing trained policy checkpoints
- StatsClient: For metrics logging and analysis
- WandbContext: For experiment tracking and visualization
- SimulationSuite: For periodic evaluation during training

**Usage**:

```bash
# Basic arena training
./tools/run.py train arena run=my_experiment

# Navigation training
./tools/run.py train navigation run=my_experiment

# Training with custom parameters
./tools/run.py train arena run=my_experiment trainer.total_timesteps=1000000

# Training with specific device and wandb settings
./tools/run.py train arena run=my_experiment system.device=cpu wandb.enabled=false
```

**Key Features**:

- Automatic worker count detection based on CPU cores
- Distributed training with PyTorch DDP
- Wandb integration for experiment tracking
- Checkpoint saving via PolicyStore
- Configurable evaluation during training

**Dependencies**:

- GPU recommended for faster training
- Optional: Stats database for metrics logging
- Optional: Wandb for experiment tracking

**Configuration**:

- Default config: `configs/train_job.yaml`
- Key parameters: `trainer.*`, `agent.*`, `env.*`

### sweep_init.py

**Purpose**: Initialize hyperparameter sweep experiments using Wandb sweeps and Metta Protein optimization.

This tool sets up systematic hyperparameter search experiments by:

- Creating Wandb sweep configurations with unique identifiers
- Initializing the Metta Protein Bayesian optimization algorithm
- Generating initial hyperparameter suggestions
- Setting up distributed sweep infrastructure

The Protein algorithm provides intelligent hyperparameter suggestions based on:

- Bayesian optimization with Gaussian processes
- Multi-objective optimization support
- Adaptive exploration/exploitation balancing
- Historical performance tracking

**Sweep initialization process**:

1. Check if sweep already exists (idempotent operation)
2. Create Wandb sweep with generated ID
3. Initialize Protein with parameter bounds
4. Generate first suggestion
5. Save configuration for distributed workers

**Usage**:

```bash
# Initialize a new sweep
./tools/sweep_init.py sweep_name=lr_search sweep_params=configs/sweep/fast.yaml

# With custom parameter configuration
./tools/sweep_init.py sweep_name=full_search sweep_params=configs/sweep/cogeval_sweep.yaml

# Distributed sweep (master node)
NODE_INDEX=0 ./tools/sweep_init.py sweep_name=distributed_exp

# Distributed sweep (worker nodes will wait)
NODE_INDEX=1 ./tools/sweep_init.py sweep_name=distributed_exp
```

**Key Features**:

- Creates Wandb sweep with unique ID
- Generates initial hyperparameter suggestions via Protein algorithm
- Supports distributed sweep initialization
- Saves sweep configuration for future runs

**Output**:

- Creates `runs/<sweep_name>/` directory
- Saves sweep metadata and Protein state
- Generates `train_config_overrides.yaml` for training

### sweep_eval.py

**Purpose**: Evaluate policies generated during hyperparameter sweeps and update Protein observations.

**Usage**:

```bash
# Evaluate a sweep run
./tools/sweep_eval.py run=<run_id> sweep_name=<sweep_name>

# With custom evaluation suite
./tools/sweep_eval.py run=<run_id> metric=navigation_score
```

**Key Features**:

- Loads policy from sweep run
- Runs configured evaluation suite
- Records metrics back to Protein for optimization
- Updates Wandb with evaluation results

**Dependencies**:

- Requires completed training run
- GPU for policy evaluation
- Access to saved checkpoints

## Evaluation Tools

### sim.py

**Purpose**: Comprehensive policy evaluation tool that runs simulation suites and exports statistics.

This tool provides comprehensive policy evaluation capabilities, allowing you to:

- Evaluate single or multiple policies in batch
- Run complete simulation suites with configurable environments
- Export detailed statistics for analysis
- Generate replays for visualization
- Support various policy selection strategies (latest, top-k by metric)

**The evaluation process**: ▸ For every requested _policy URI_ ▸ choose the checkpoint(s) according to selector/metric ▸
run the configured `SimulationSuite` ▸ export the merged stats DB if an output URI is provided ▸ output JSON results for
programmatic processing

**Integration points**:

- CheckpointManager: For loading trained policies
- SimulationSuite: For running evaluation scenarios
- StatsDB: For storing and exporting metrics
- StatsClient: For real-time metrics streaming

**Usage**:

```bash
# Eval a single policy
./tools/run.py eval navigation policy_uri=s3://my-bucket/checkpoints/experiment_001/experiment_001:v12.pt

# Eval with arena recipe
./tools/run.py eval arena policy_uri=s3://my-bucket/checkpoints/experiment_001/experiment_001:v12.pt

# Eval with specific policy from file
./tools/run.py eval arena policy_uri=file://./train_dir/my_run/checkpoints/my_run:v12.pt

# Eval using a remote checkpoint stored on S3
./tools/run.py eval navigation policy_uri=s3://team-checkpoints/project/my_run/checkpoints/my_run:v0.pt
```

**Key Features**:

- Supports multiple policy evaluation in one run
- Flexible policy selection (latest, top-k by metric)
- Replay generation for visualization
- Stats database export for analysis
- JSON output for programmatic use

**Output Format**:

```json
{
  "simulation_suite": "navigation",
  "policies": [
    {
      "policy_uri": "s3://my-bucket/checkpoints/abc123/abc123:v5.pt",
      "checkpoints": [
        {
          "name": "checkpoint_1000",
          "uri": "s3://my-bucket/checkpoints/abc123/abc123:v5.pt",
          "metrics": {
            "reward_avg": 15.3
          }
        }
      ]
    }
  ]
}
```

### analyze.py

**Purpose**: Generate detailed analysis reports from evaluation results, including performance metrics and behavior
analysis.

This tool generates comprehensive analysis reports from policy evaluation data, providing insights into agent behavior,
performance metrics, and learning progress.

**Key features**:

- Statistical analysis of agent performance across episodes
- Behavior pattern detection and visualization
- Comparative analysis between checkpoints
- Export to multiple formats (HTML, JSON, PDF)
- Integration with CheckpointManager for policy metadata

**The analysis pipeline**:

1. Load policy from checkpoints using specified selector
2. Retrieve evaluation statistics from connected databases
3. Generate statistical summaries and visualizations
4. Export analysis results to specified output path

**Usage**:

```bash
# Analyze arena evaluation results
./tools/run.py analyze arena eval_db_uri=./train_dir/eval_experiment/stats.db

# Analyze navigation evaluation results
./tools/run.py analyze navigation eval_db_uri=./train_dir/eval_experiment/stats.db

# Analyze with specific evaluation database
./tools/run.py analyze arena eval_db_uri=wandb://artifacts/navigation_db
```

**Dependencies**:

- Requires completed evaluation data
- Access to stats database
- CheckpointManager for loading policies

## Visualization Tools

### renderer.py

**Purpose**: Real-time visualization of agent behavior with ASCII or Miniscope rendering.

This tool provides interactive visualization of policies (random, heuristic, or trained) running in MettaGrid
environments with support for multiple rendering backends.

**Key features**:

- Multiple policy types: random, simple heuristic, or trained neural networks
- Multiple rendering modes: ASCII (NetHack-style), Miniscope (rich visuals)
- Real-time performance metrics display
- Support for multi-agent environments
- Configurable simulation speed
- Action validation for trained policies

**Policy types**:

- `random`: Generates valid random actions for baseline comparison
- `simple`: Heuristic policy with movement preferences (60% cardinal, 20% diagonal, etc.)
- `trained`: Loads policies from checkpoints with automatic action validation

**Rendering modes**:

- `human`: ASCII renderer with NetHack-style display (default)
- `miniscope`: Rich graphical renderer with sprites and effects
- `nethack`: Alias for human mode

**Usage**:

```bash
# Visualize random policy
./tools/renderer.py renderer_job.policy_type=random

# Visualize trained policy
./tools/renderer.py renderer_job.policy_type=trained policy_uri="s3://my-bucket/checkpoints/experiment/experiment:v20.pt"

# Custom environment with multiple agents
./tools/renderer.py env=mettagrid/multiagent renderer_job.num_agents=4

# Slow motion visualization
./tools/renderer.py renderer_job.sleep_time=0.5

# Miniscope renderer
./tools/renderer.py renderer_job.renderer_type=miniscope
```

### replay.py

**Purpose**: Generate replay files for detailed post-hoc analysis in MettaScope.

**Usage**:

```bash
# Generate replay for a policy
./tools/run.py replay arena policy_uri=s3://my-bucket/checkpoints/abc123/abc123:v5.pt

# Generate replay with navigation environment
./tools/run.py replay navigation policy_uri=s3://my-bucket/checkpoints/abc123/abc123:v5.pt

# Generate replay from local checkpoint
./tools/run.py replay arena policy_uri=file://./train_dir/my_run/checkpoints/my_run:v12.pt
```

**Key Features**:

- Generates `.replay` files for MettaScope
- Automatic browser launch on macOS
- Local server for replay viewing
- Configurable simulation parameters

### play.py

**Purpose**: Interactive gameplay interface allowing humans to control Metta agents.

**Usage**:

```bash
# Start interactive session
./tools/run.py play arena

# Interactive play with specific policy
./tools/run.py play arena policy_uri=s3://my-bucket/checkpoints/my_experiment/my_experiment:v20.pt

# Interactive navigation environment
./tools/run.py play navigation policy_uri=file://./train_dir/nav_experiment/checkpoints/nav_experiment:v8.pt
```

**Key Features**:

- WebSocket-based real-time control
- Browser-based interface via MettaScope
- Human-in-the-loop testing
- Recording of human gameplay

### dashboard.py

**Purpose**: Generate dashboard data for web-based visualization of training runs and evaluations.

This tool aggregates metrics and statistics from multiple training runs and evaluations to create a unified dashboard
view accessible via the Metta Observatory web interface.

**Key features**:

- Aggregates metrics across multiple training runs
- Generates interactive visualization data
- Supports both local and S3 output destinations
- Provides direct links to web-based dashboard viewer
- Compatible with Metta Observatory for real-time monitoring

**The dashboard generation process**:

1. Collect metrics from specified training runs
2. Aggregate statistics across policies and checkpoints
3. Generate JSON data structure for web visualization
4. Upload to S3 or save locally with viewer URL

**Usage**:

```bash
# Generate dashboard and upload to S3
./tools/dashboard.py dashboard.output_path=s3://my-bucket/dashboards/experiment.json

# Generate local dashboard file
./tools/dashboard.py dashboard.output_path=./local_dashboard.json

# Custom dashboard configuration
./tools/dashboard.py dashboard.policies=["run1", "run2"] dashboard.metrics=["reward", "success_rate"]
```

## Map Tools

### map/gen.py

**Purpose**: Generate MettaGrid maps from configuration files using various procedural algorithms.

This tool creates game maps using different generation algorithms including:

- WaveFunctionCollapse (WFC): Pattern-based generation from examples
- ConvChain: Convolutional pattern generation
- Random: Stochastic map generation with constraints
- Template-based: Fixed layouts with random variations

**Key features**:

- Batch generation support for creating map datasets
- Multiple output formats (YAML, visual preview)
- S3 and local file system support
- Configuration override system for parameter tuning
- Automatic map validation and normalization

**The generation pipeline**:

1. Load map configuration (Hydra or OmegaConf format)
2. Initialize chosen generation algorithm
3. Generate map according to constraints
4. Validate generated content
5. Save or display results

**Output modes**:

- Save to file/S3 with automatic format detection
- Display in MettaScope web viewer
- ASCII terminal display
- PIL image popup

**Usage**:

```bash
# Generate and display a single map
  ./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/maze_9x9.yaml

# Save map to file
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/wfc_dungeon.yaml --output-uri=./dungeon.yaml

# Generate 100 maps to S3
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/random.yaml --output-uri=s3://bucket/maps/ --count=100

# Override generation parameters
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/base.yaml "width=50 height=50 density=0.7"

# Different display modes
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py map_config.yaml --show-mode=ascii # Terminal display
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py map_config.yaml --show-mode=PIL  # Image popup
```

### map/gen_scene.py

**Purpose**: Generate maps from scene configuration files with specified dimensions.

**Usage**:

```bash
# Generate from scene
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen_scene.py scenes/wfc/blob.yaml 32 32

# With overrides
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen_scene.py scenes/convchain/maze.yaml 64 64 "seed=42"

# Different display mode
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen_scene.py scenes/test/grid.yaml 16 16 --show-mode=ascii
```

**Key Features**:

- Dynamic size specification
- Scene template system
- Pattern-based generation

### map/view.py

**Purpose**: View stored maps from various sources (local files, S3, etc.).

**Usage**:

```bash
# View a specific map
./packages/mettagrid/python/src/mettagrid/mapgen/tools/view.py ./my_map.yaml

# View random map from directory
./packages/mettagrid/python/src/mettagrid/mapgen/tools/view.py s3://bucket/maps/

# ASCII display
./tools/map/view.py ./map.yaml --show-mode=ascii
```

**Key Features**:

- Auto-detection of directories vs files
- Random map selection from directories
- Multiple visualization modes

### map/normalize_ascii_map.py

**Purpose**: Normalize ASCII map characters to ensure consistency across different encodings.

**Usage**:

```bash
# Print normalized map
./tools/map/normalize_ascii_map.py map.txt

# Normalize in-place
./tools/map/normalize_ascii_map.py map.txt --in-place
```

**Key Features**:

- Unicode normalization
- Character substitution rules
- Preserves map structure

### map/normalize_scene_patterns.py

**Purpose**: Normalize patterns in WFC/ConvChain scene configuration files.

**Usage**:

```bash
# Print normalized config
./tools/map/normalize_scene_patterns.py scene.yaml

# Update file in-place
./tools/map/normalize_scene_patterns.py scene.yaml --in-place
```

## Utility Tools

### validate_config.py

**Purpose**: Load and validate Hydra configuration files, useful for debugging config issues.

This tool helps developers understand and troubleshoot complex Hydra configurations by loading, resolving, and
displaying them in a readable format. It's particularly useful for debugging configuration composition and interpolation
issues.

**Key features**:

- Loads any Hydra configuration with proper composition
- Handles complex configuration groups (env, trainer, agent, etc.)
- Attempts to resolve interpolations where possible
- Provides fallback for configs with missing dependencies
- Supports both simple configs and complex multi-file compositions

**The validation process**:

1. Load configuration using Hydra's composition system
2. Apply necessary overrides for missing required fields
3. Attempt to resolve all interpolations
4. Display final configuration in YAML format
5. Report any issues or unresolvable references

**Usage**:

```bash
# Validate environment configuration
./tools/validate_config.py configs/env/mettagrid/navigation.yaml

# Validate trainer configuration
./tools/validate_config.py trainer/trainer.yaml

# Validate complex composed configuration
./tools/validate_config.py configs/train_job.yaml

# Check sweep configuration
./tools/validate_config.py configs/sweep/fast.yaml
```

**Note**: Currently this script prints the configuration, but future versions will add schema validation and constraint
checking.

### stats_duckdb_cli.py

**Purpose**: Interactive DuckDB CLI for exploring evaluation statistics databases.

This tool provides a convenient interface for analyzing evaluation metrics stored in DuckDB databases, with automatic
downloading from remote sources (Wandb, S3).

**Key features**:

- Automatic download from wandb:// or s3:// URIs
- Direct DuckDB CLI access for SQL queries
- Temporary local caching of remote databases
- Full SQL support for complex analytics
- Integration with evaluation pipeline outputs

**Common query patterns**:

- Agent performance metrics aggregation
- Episode-level statistics analysis
- Policy comparison queries
- Temporal performance trends
- Multi-agent coordination metrics

**Database schema typically includes**:

- agent_metrics: Per-step agent observations and rewards
- episode_summary: Aggregated episode statistics
- policy_metadata: Policy configuration and versioning
- simulation_config: Environment and task settings

**Usage**:

```bash
# Open stats from Wandb
./tools/stats_duckdb_cli.py +eval_db_uri=wandb://stats/navigation_eval_v2

# Open stats from S3
./tools/stats_duckdb_cli.py +eval_db_uri=s3://my-bucket/evaluations/experiment_001.db
```

**Example Queries**:

```sql
-- View available tables
.tables

-- Get average rewards by policy
SELECT policy_name, AVG(value) as avg_reward
FROM agent_metrics
WHERE metric = 'reward'
GROUP BY policy_name;

-- Analyze episode lengths
SELECT policy_name, episode, COUNT(*) as steps
FROM agent_metrics
GROUP BY policy_name, episode;

-- Schema info
.schema agent_metrics
```

### upload_map_imgs.py

**Purpose**: Batch upload map visualization images to S3 for public access.

**Usage**:

```bash
# Dry run to see what would be uploaded
./tools/upload_map_imgs.py --dry-run

# Upload all images in current directory
./tools/upload_map_imgs.py
```

**Key Features**:

- Auto-detection of image files
- Proper MIME type handling
- S3 public bucket upload
- Batch processing

### dump_src.py

**Purpose**: Dump source code files for LLM context or analysis.

**Usage**:

```bash
# Dump all Python files
./tools/dump_src.py . --extensions .py

# Dump specific directories
./tools/dump_src.py metta/agent metta/rl --extensions .py .yaml

# All files in a directory
./tools/dump_src.py configs/env/
```

**Output Format**:

```
<file: path/to/file.py>
... file contents ...
</file>
```

### autotune.py

**Purpose**: Auto-tune vectorization parameters for optimal performance using pufferlib.

**Usage**:

```bash
# Run autotuning
./tools/autotune.py
```

**Key Features**:

- Finds optimal batch size
- Determines max environments
- Memory usage optimization
- Performance benchmarking

## Workflow Examples

### Training and Evaluation Pipeline

```bash
# 1. Train a policy
./tools/run.py train navigation run=nav_experiment_001

# 2. Eval the trained policy
./tools/run.py eval navigation policy_uri=s3://my-bucket/checkpoints/nav_experiment_001/nav_experiment_001:v8.pt

# 3. Analyze results
./tools/run.py analyze navigation eval_db_uri=./train_dir/eval_nav_experiment_001/stats.db

# 4. Interactive play with trained policy
./tools/run.py play navigation policy_uri=s3://my-bucket/checkpoints/nav_experiment_001/nav_experiment_001:v8.pt
```

### Hyperparameter Sweep Workflow

```bash
# 1. Initialize sweep
./tools/sweep_init.py sweep_name=hyperparam_search_001 \
  sweep_params=configs/sweep/navigation_sweep.yaml

# 2. Training happens automatically via recipe system

# 3. Evaluate sweep runs
./tools/sweep_eval.py run=<run_id> sweep_name=hyperparam_search_001

# 4. Interactive play with best policy
./tools/run.py play arena \
   policy_uri="s3://my-bucket/checkpoints/sweeps/hyperparam_search_001/hyperparam_search_001:v42.pt"
```

### Map Development Workflow

```bash
# 1. Generate a new map
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/template.yaml \
  --output-uri=./my_map.yaml "seed=42"

# 2. View and iterate
./tools/map/view.py ./my_map.yaml

# 3. Normalize if needed
./tools/map/normalize_ascii_map.py ./my_map.yaml --in-place

# 4. Test with renderer
./tools/renderer.py env.game.map=@file://./my_map.yaml
```

## Environment Variables

Key environment variables used by tools:

- `RANK`: Distributed training rank
- `LOCAL_RANK`: Local GPU rank
- `NODE_INDEX`: Node index in multi-node setup
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `WANDB_API_KEY`: Wandb authentication
- `AWS_PROFILE`: AWS credentials for S3 access

## Common Issues and Solutions

### GPU Memory Issues

```bash
# Use CPU for testing
./tools/run.py train arena run=cpu_test system.device=cpu

# Reduce training time for quick testing
./tools/run.py train arena run=quick_test trainer.total_timesteps=10000
```

### Database Access

```bash
# For local testing without external services
./tools/run.py train arena run=local_test wandb.enabled=false system.device=cpu
```

## Best Practices

1. **Always validate configs** before long-running experiments
2. **Use meaningful run names** for easy identification
3. **Export evaluation data** to S3 for persistence
4. **Monitor GPU memory** usage during training
5. **Use sweep tools** for systematic hyperparameter search
6. **Generate replays** for debugging unexpected behaviors
7. **Document custom configurations** in version control
