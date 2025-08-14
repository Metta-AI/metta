# Experiments

A framework for reproducible experiments with Metta. This includes:
- Systematic launching of training jobs for MARL research experiments
- Full configuration management using Python config objects
- Automatic YAML serialization for Skypilot job transfer
- Notebook infrastructure for analysis and visualization (marimo-based)

Note: The experiment launching and notebook systems are not yet connected. Integration will be added in a future update.

## Quick Start

### Create and Launch Experiments

```bash
# Create and launch arena experiment
uv run experiments/recipes/arena_experiment.py

# With custom parameters
uv run experiments/recipes/arena_experiment.py --gpus=4 --nodes=2 --launch
```

## Structure

```
experiments/
├── experiment.py             # Base Experiment class
├── training_job.py          # TrainingJob and TrainingJobConfig
├── skypilot_job_config.py   # Infrastructure/deployment configuration  
├── training_run_config.py   # Training run configuration with YAML serialization
├── wandb_service.py         # WandB data fetching utilities  
├── skypilot_service.py      # Sky job monitoring with config file transfer
├── runner.py                # CLI runner with typer integration
├── recipes/                 # Experiment implementations
│   └── arena_experiment.py
├── notebooks/               # Notebook infrastructure (marimo/jupyter)
├── marimo/                 # Marimo-based notebooks
└── scratch/                 # Working directory (git-ignored)
```

## Configuration System

The experiment framework uses a clean configuration architecture:

### Training Configuration

The configuration system cleanly separates infrastructure concerns from training parameters:

```python
from experiments.skypilot_job_config import SkypilotJobConfig
from experiments.training_run_config import TrainingRunConfig
from experiments.training_job import TrainingJobConfig
from metta.rl.trainer_config import TrainerConfig, OptimizerConfig, PPOConfig

# 1. Infrastructure configuration (Skypilot)
skypilot = SkypilotJobConfig(
    gpus=4,
    nodes=2,
    spot=False,
    max_runtime_hours=2.0,
)

# 2. Training configuration
trainer = TrainerConfig(
    total_timesteps=10_000_000,
    num_workers=4,
    batch_size=4096,
    optimizer=OptimizerConfig(
        type="adam",
        learning_rate=0.0003,
    ),
    ppo=PPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
    ),
)

training = TrainingRunConfig(
    curriculum="env/mettagrid/curriculum/arena/learning_progress",
    agent_config="latent_attn_tiny",  # References configs/agent/latent_attn_tiny.yaml
    wandb_entity="my-org",
    wandb_project="my-project",
    trainer=trainer,
)

# 3. Combine into complete job configuration
job_config = TrainingJobConfig(
    skypilot=skypilot,
    training=training,
)
```

### YAML Serialization

The training configuration is automatically serialized to YAML for transfer to remote training:

```python
# Serialize training config to YAML file
yaml_path, full_config = training.serialize_to_yaml_file()
```

This generates a complete Hydra-compatible YAML configuration that includes:
- Agent configuration references
- Trainer settings with all nested objects
- WandB configuration
- All hyperparameters

Infrastructure settings (from SkypilotJobConfig) are passed as command-line arguments to maintain clean separation of concerns.

### Clean Separation of Concerns

The three-level configuration hierarchy provides clear separation:

1. **SkypilotJobConfig**: Infrastructure/deployment settings
   - GPUs, nodes, spot instances, runtime limits
   - Passed as CLI arguments to launch.py

2. **TrainingRunConfig**: Training run configuration
   - Agent, curriculum, wandb, trainer settings
   - Serialized to YAML and transferred via file mounts

3. **TrainingJobConfig**: Complete job specification
   - Combines both configs for a full job definition
   - Used by the experiment framework

### How It Works

1. **Local**: Create `TrainingJobConfig` with `TrainerConfig` objects
2. **Serialization**: Config is serialized to a YAML file
3. **Transfer**: Skypilot mounts the YAML file to `/tmp/metta_train_config.yaml` on remote
4. **Remote**: Training script loads config via Hydra's `--config-path` and `--config-name`

This approach:
- ✅ Avoids command-line length limitations
- ✅ Provides type-safe configuration with Pydantic models
- ✅ Maintains clean separation between infrastructure and training configs
- ✅ Enables reproducible experiments with versioned configs

## Service Architecture for Interactive Analysis

The `SkypilotService` and `WandbService` classes are designed as service layers that will enable rich interactive analysis and monitoring:

### Current Capabilities
- Query job status and metadata from Skypilot
- Retrieve wandb run information and metrics
- Track launched experiments and their configurations
- Validate and preflight check configurations

### Future Integration Points
These services are architected to support interactive notebooks and dashboards:

- **Jupyter/Marimo notebooks**: Query training status, visualize metrics, compare runs
- **Plotly dashboards**: Real-time training monitoring with interactive plots  
- **Streamlit apps**: Custom experiment management interfaces
- **Automated reporting**: Generate training reports and summaries

The service pattern provides a clean API boundary between the experiment framework and visualization/analysis tools. This makes it easy to build rich interactive experiences on top of the training infrastructure, such as:

```python
# Future notebook usage example
from experiments.wandb_service import get_wandb_service
from experiments.skypilot_service import get_skypilot_service

# Query running experiments
sky_service = get_skypilot_service()
jobs = sky_service.get_running_jobs()

# Get metrics for analysis
wandb_service = get_wandb_service()
metrics = wandb_service.get_run_metrics(run_id)

# Visualize in interactive dashboard
import plotly.express as px
fig = px.line(metrics, x='step', y='reward')
fig.show()
```

This design enables researchers to build custom analysis workflows while the core experiment framework handles the complexity of distributed training infrastructure.

