# Learning Progress Arena Experiment

This experiment compares three curriculum approaches for arena environments:

1. **Learning Progress Curriculum** - Adaptively samples tasks based on learning progress
2. **Random Curriculum** - Uniform random sampling across the same 12 arena tasks
3. **Basic Arena Baseline** - Training only on basic_easy arena environment

## Experiment Structure

### Files Created

#### Configurations
- `configs/env/mettagrid/curriculum/arena_random.yaml` - Random curriculum using the same 12 arena tasks
- `configs/sweep/learning_progress_arena.yaml` - Protein hyperparameter sweep configuration
- `configs/sweep_job_learning_progress.yaml` - Sweep job configuration
- `configs/user/learning_progress_experiment.yaml` - Learning progress experiment config
- `configs/user/random_curriculum_experiment.yaml` - Random curriculum experiment config
- `configs/user/basic_arena_experiment.yaml` - Basic arena baseline experiment config

#### Scripts
- `recipes/learning_progress_arena_experiment.sh` - Main experiment script

## Running the Experiment

### Quick Start

```bash
# Run the complete experiment
./recipes/learning_progress_arena_experiment.sh
```

### Manual Steps

If you prefer to run steps manually:

#### 1. Hyperparameter Sweep

```bash
./devops/skypilot/launch.py sweep \
    --gpus=4 \
    --nodes=8 \
    --no-spot \
    --config configs/sweep_job_learning_progress.yaml \
    sweep_name="$USER.learning_progress_arena.sweep" \
    trainer.total_timesteps=1_000_000_000
```

#### 2. Comparison Experiments (after sweep completes)

```bash
# Learning Progress Curriculum (with best hyperparameters)
./devops/skypilot/launch.py train \
    --gpus=4 \
    --nodes=8 \
    --no-spot \
    run="$USER.learning_progress_arena.learning_progress" \
    --config configs/user/learning_progress_experiment.yaml \
    trainer.total_timesteps=1_000_000_000 \
    ema_timescale=0.001 \
    progress_smoothing=0.05 \
    num_active_tasks=16 \
    rand_task_rate=0.25 \
    sample_threshold=10 \
    memory=25

# Random Curriculum Baseline
./devops/skypilot/launch.py train \
    --gpus=4 \
    --nodes=8 \
    --no-spot \
    run="$USER.learning_progress_arena.random" \
    --config configs/user/random_curriculum_experiment.yaml \
    trainer.total_timesteps=1_000_000_000

# Basic Arena Baseline
./devops/skypilot/launch.py train \
    --gpus=4 \
    --nodes=8 \
    --no-spot \
    run="$USER.learning_progress_arena.basic" \
    --config configs/user/basic_arena_experiment.yaml \
    trainer.total_timesteps=1_000_000_000
```

## Experiment Details

### Learning Progress Hyperparameters

The sweep optimizes these hyperparameters:

- `ema_timescale` (0.0001 - 0.01): Exponential moving average timescale
- `progress_smoothing` (0.01 - 0.2): Progress smoothing parameter
- `num_active_tasks` (8 - 24): Number of active tasks to sample from
- `rand_task_rate` (0.1 - 0.5): Rate of random task selection
- `sample_threshold` (5 - 20): Minimum samples before using learning progress
- `memory` (15 - 50): Memory window for task outcomes

### Arena Tasks

The learning progress and random curricula use these 12 arena tasks:

- `/env/mettagrid/arena/basic`
- `/env/mettagrid/arena/basic_easy`
- `/env/mettagrid/arena/basic_easy_shaped`
- `/env/mettagrid/arena/combat`
- `/env/mettagrid/arena/combat_easy`
- `/env/mettagrid/arena/combat_easy_shaped`
- `/env/mettagrid/arena/advanced`
- `/env/mettagrid/arena/advanced_easy`
- `/env/mettagrid/arena/advanced_easy_shaped`
- `/env/mettagrid/arena/tag`
- `/env/mettagrid/arena/tag_easy`
- `/env/mettagrid/arena/tag_easy_shaped`

### Evaluation

All experiments use the arena evaluation suite (`/sim/arena`) which includes:
- `arena/basic`
- `arena/combat`
- `arena/advanced`
- `arena/tag`

## Monitoring

Monitor experiment progress at: https://wandb.ai/metta-research/learning-progress-sweep

Look for runs with tags: `["learning_progress_arena_comparison"]`

## Expected Results

The experiment should show:

1. **Learning Progress Curriculum**: Should adaptively focus on tasks where the agent is making progress
2. **Random Curriculum**: Should provide uniform exploration across all tasks
3. **Basic Arena**: Should show performance on the single basic_easy environment

Compare final rewards across all three approaches to determine which curriculum strategy is most effective.

## Troubleshooting

### Common Issues

1. **Sweep not completing**: Check wandb for sweep status and manually monitor progress
2. **Hyperparameter extraction**: You may need to manually check wandb for best hyperparameters
3. **Resource allocation**: Adjust `--gpus` and `--nodes` based on available resources

### Manual Hyperparameter Update

After the sweep completes, update the hyperparameters in the learning progress launch command with the best values found by the sweep.

## Notes

- Each experiment runs for 1 billion timesteps
- The sweep uses 20 random samples before starting Bayesian optimization
- All experiments use the same evaluation suite for fair comparison
- Results are logged to wandb for easy comparison and analysis
