# Doxascope

**Doxascope** is a neural probe for investigating the internal representations of reinforcement learning agents. It
trains a neural network to predict an agent's relative past and future grid positions from its LSTM memory states,
revealing what spatial information is encoded in the agent's internal representations.

## Overview

The core idea: if a network can accurately predict where an agent was or will be based solely on its LSTM hidden and
cell states, then those states must encode spatial trajectory information. By analyzing prediction accuracy across
different timesteps and spatial offsets, we can visualize what the agent "knows" or "remembers" about its movement.

### Module Structure

- **`doxascope_data.py`**: Data logging (`DoxascopeLogger`) and preprocessing utilities
- **`doxascope_network.py`**: PyTorch network (`DoxascopeNet`) and training pipeline (`DoxascopeTrainer`)
- **`doxascope_analysis.py`**: Cross-run and cross-policy comparison functions
- **`doxascope_plots.py`**: Visualization utilities (training curves, accuracy heatmaps)
- **`cli.py`**: Unified interactive CLI with 4 main commands
- **`test_doxascope.py`**: Comprehensive test suite

### Network Architecture

The `DoxascopeNet` processes LSTM states through:

1. **Parallel Processors**: Separate MLPs for hidden and cell states
2. **Feature Fusion**: Concatenates processed features
3. **Main Network**: Shared MLP trunk
4. **Multi-Head Outputs**: Separate classification heads for each timestep

Each prediction head uses **Manhattan distance classification**: for timestep offset `k`, the network classifies
relative position `(dr, dc)` into one of the possible cells reachable within `k` steps.

### Core Workflow

1.  **Collect Data**: Enable doxascope during policy evaluation to log LSTM states and positions
2.  **Train Model**: Train network on collected data with automatic train/val/test splitting
3.  **Analyze Results**: Generate accuracy plots, heatmaps, and per-timestep visualizations
4.  **Compare Runs**: Overlay multiple training runs to track improvement over time

## Directory Structure

All doxascope data lives in `train_dir/doxascope/` (gitignored to avoid committing large data files):

```
train_dir/doxascope/
├── raw_data/<policy_name>/
│   └── doxascope_data_<simulation_id>.json     # Logged LSTM states and positions
├── results/<policy_name>/<run_name>/
│   ├── best_model.pth                          # Best checkpoint (full)
│   ├── best_model.state_dict.pth               # State dict only
│   ├── training_history.csv                     # Loss and accuracy per epoch
│   ├── test_results.json                        # Test accuracy per timestep
│   ├── test_results_baseline.json              # Random baseline results (if enabled)
│   ├── preprocessed_data/
│   │   ├── train.npz                            # Cached training data
│   │   ├── val.npz                              # Cached validation data
│   │   ├── test.npz                             # Cached test data
│   │   └── *_baseline.npz                       # Baseline datasets (randomized inputs)
│   └── analysis/
│       ├── training_history.png
│       ├── multistep_accuracy_comparison.png
│       └── acc_heatmap_t[±k].png               # Per-timestep accuracy heatmaps
└── sweeps/<policy_name>/<sweep_name>/
    ├── sweep_summary.csv                        # All trials sorted by performance
    └── all_trial_results.json                   # Raw trial data
```

## Usage

The Doxascope CLI provides an **interactive mode** for all commands. Simply run without arguments and follow the
prompts:

```bash
uv run doxascope  # Interactive menu to select command
```

Or specify a command directly:

```bash
uv run doxascope <command> [options]
```

### 1. Data Collection

#### Using the Doxascope Recipe (Recommended)

The easiest way to collect data is via the interactive CLI:

```bash
uv run doxascope collect
```

This will prompt you for:

- Policy URI to evaluate
- Number of simulations to run

Alternatively, use the doxascope recipe directly:

```bash
uv run ./tools/run.py experiments.recipes.doxascope.evaluate policy_uri=<path> num_simulations=10
```

#### Manual Collection

Enable doxascope during any evaluation by adding `doxascope_enabled=true`:

```bash
uv run ./tools/run.py experiments.recipes.arena.evaluate policy_uri=<path> doxascope_enabled=true
```

**Data Logging**: The `DoxascopeLogger` automatically detects LSTM states from:

- TensorDict (per-forward recurrent state)
- Policy state buffers (`policy.state.lstm_h/lstm_c`)
- Component buffers (`components['lstm_reset']`)

**Important**: Doxascope currently only supports **single-environment logging**. The doxascope recipe defaults to
single-environment evaluation for this reason. Multi-environment setups are not currently supported.

Raw location and LSTM data is saved to `train_dir/doxascope/raw_data/<policy_name>/`. Run multiple simulations to
accumulate data.

### 2. Training the Doxascope Network

#### Interactive Training

The simplest way to train is via interactive mode:

```bash
uv run doxascope train
```

This will:

1. Show available policies with data
2. Let you configure prediction timesteps (past/future)
3. Show and let you modify all training settings
4. Train the main model + optional baseline

#### Non-Interactive Training

```bash
uv run doxascope train <policy_name> [options]
```

#### Key Configuration

**Prediction Settings:**

- `--num-future-timesteps <n>`: Number of future positions to predict (default: 1)
- `--num-past-timesteps <n>`: Number of past positions to predict (default: 0)

**Training Settings:**

- `--batch-size <n>`: Batch size (default: 32)
- `--learning-rate <lr>`: Learning rate (default: 0.001)
- `--num-epochs <n>`: Maximum epochs (default: 100)
- `--patience <n>`: Early stopping patience (default: 10)
- `--device <device>`: 'cpu', 'cuda', or 'auto' (default: 'auto')

**Data Splits:**

- `--test-split <ratio>`: Test set proportion (default: 0.15)
- `--val-split <ratio>`: Validation set proportion (default: 0.15)

**Model Architecture:**

- `--hidden_dim <n>`: Hidden dimension (default: 512)
- `--dropout_rate <rate>`: Dropout rate (default: 0.4)
- `--activation_fn <fn>`: 'relu', 'silu', or 'gelu' (default: 'gelu')
- `--main_net_depth <n>`: Main network depth (default: 3)
- `--processor_depth <n>`: State processor depth (default: 1)

**Baseline Comparison:**

- `--train-random-baseline` / `--no-train-random-baseline`: Train baseline model with randomized inputs (default:
  enabled)

The baseline uses the same architecture and labels, but with **randomized LSTM inputs**, establishing a performance
floor. This helps determine if the network is truly learning from memory content vs. just exploiting label distribution
biases.

### 3. Analysis and Comparison

#### Compare Command

The `compare` command visualizes performance across runs and policies:

**Interactive Mode:**

```bash
uv run doxascope compare
```

Choose between:

1. Compare multiple runs within a single policy (tracks improvement over time)
2. Compare latest runs across multiple policies (cross-policy comparison)

**Non-Interactive:**

```bash
# Compare runs within one policy
uv run doxascope compare <policy_name>

# Compare latest runs across policies
uv run doxascope compare <policy_1> <policy_2> <policy_3>
```

#### Visualizations Generated

Training automatically generates analysis plots in `<run_dir>/analysis/`:

1. **`training_history.png`**: Loss and accuracy curves over training epochs
2. **`multistep_accuracy_comparison.png`**: Test accuracy per timestep (includes baseline if enabled)
3. **`acc_heatmap_t[±k].png`**: Spatial accuracy heatmaps for each prediction timestep
   - **Color**: Prediction accuracy (0-100%)
   - **Text annotation**: Sample count for that relative position
   - **Grid**: Centered at (0,0), shows all Manhattan-reachable positions

The heatmaps reveal spatial biases in what the agent encodes—e.g., higher accuracy for forward positions may indicate
the agent's memory prioritizes where it's heading.

### 4. Hyperparameter Sweep

Find optimal configurations through random search:

#### Interactive Mode

```bash
uv run doxascope sweep
```

Guides you through:

1. Policy selection
2. Prediction timestep configuration
3. Sweep type (hyperparameter vs. architecture)
4. Sweep parameters (num configs, epochs, patience)

#### Non-Interactive Mode

```bash
uv run doxascope sweep <policy_name> <num_future_timesteps> [options]
```

**Options:**

- `--num-past-timesteps <n>`: Past steps to predict (default: 0)
- `--num-configs <n>`: Number of random configs to test (default: 30)
- `--max-epochs <n>`: Max epochs per trial (default: 50)
- `--patience <n>`: Early stopping patience per trial (default: 10)
- `--sweep-type <type>`: `hyper` or `arch` (default: `hyper`)

#### Search Spaces

**Hyperparameter Sweep (`--sweep-type hyper`):**

- `hidden_dim`: [128, 256, 512]
- `dropout_rate`: [0.2, 0.4, 0.6]
- `lr`: [0.0001, 0.0005, 0.001]
- Fixed: `activation_fn='gelu'`, `main_net_depth=3`, `processor_depth=1`

**Architecture Sweep (`--sweep-type arch`):**

- `activation_fn`: ['gelu', 'relu', 'silu']
- `main_net_depth`: [1, 2, 3]
- `processor_depth`: [1, 2]
- Fixed: `hidden_dim=512`, `dropout_rate=0.4`, `lr=0.0007`

Results saved to `train_dir/doxascope/sweeps/<policy_name>/<sweep_name>/`:

- `sweep_summary.csv`: All trials ranked by validation accuracy
- `all_trial_results.json`: Full trial data including configs and histories

### Data Requirements

Minimum recommended data for meaningful results:

- **Training samples**: ~1000+ per timestep prediction head
- **Simulations**: 10+ simulation runs of 50+ steps each
- **Movement diversity**: Agent should explore various relative positions

Monitor `preprocessed_data/*.npz` file sizes—expect several MB for adequate training.

## Technical Details

### Manhattan Distance Encoding

For a prediction offset of `k` timesteps, the agent's relative position `(dr, dc)` must satisfy:

```
|dr| + |dc| ≤ |k|
```

This creates `2k² + 2k + 1` reachable cells (no diagonal movement). Each cell is assigned a unique class ID based on a
canonical sorted ordering: `(dr, dc)` sorted by row then column.

**Example** (k=1):

- Classes: 5 cells
- Positions: [(-1,0), (0,-1), (0,0), (0,1), (1,0)]
- Center (0,0) → class_id=2

**Example** (k=2):

- Classes: 13 cells
- Includes all cells within Manhattan distance 2

### Data Splitting

Training uses **file-level splitting** to prevent data leakage:

1. All JSON files are shuffled (seeded for reproducibility)
2. Files are distributed to train/val/test based on file size to balance data volume
3. Each agent trajectory stays entirely within one split

This ensures the model generalizes to new simulation runs, not just new timesteps from seen runs.

### Network Details

- **Input**: Concatenated LSTM hidden and cell states `[h; c]` (typically 512-1024 dim)
- **Processors**: Separate MLPs reduce `h` and `c` to lower-dimensional representations
- **Main network**: Processes concatenated features through shared MLP trunk
- **Output heads**: One linear classifier per timestep, each with its own class count

Loss is averaged across all prediction heads using standard cross-entropy.

## Troubleshooting

**"No recurrent state found"**

- Doxascope requires LSTM-based policies. Ensure your policy uses recurrent components.
- Check that the policy exposes LSTM state via TensorDict, `policy.state`, or component buffers.

**Low accuracy near baseline**

- Insufficient training data—collect more simulations
- Agent movement may be too regular (not exploring diverse positions)
- LSTM states may genuinely not encode spatial information. Probably untrained or poor performing policy (this is a
  valid finding!)
