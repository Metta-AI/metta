# Metta AI - Architecture Quick Reference

## At a Glance

**Metta AI** is a multi-agent RL system studying cooperation emergence. Architecture: 
- **Recipes** define experiment configurations
- **Tools** (TrainTool, EvaluateTool, etc.) execute workflows
- **Trainer** orchestrates training with pluggable components
- **MettaGrid** (C++ env) runs fast simulations
- **Checkpoints** store policies with versioned URIs

---

## Directory Structure

```
metta/                          # Main Python package
├── tools/                       # train.py, eval.py, play.py
├── rl/                          # Trainer, losses, training components
├── sim/                         # Simulation, evaluation
├── eval/                        # Analysis, eval service
└── common/tool/                 # run_tool.py (main CLI entry)

agent/                           # Neural networks
├── policy.py                    # Base Policy interface
└── policies/                    # ViT, Fast, LSTM, etc.

packages/mettagrid/             # C++ environment
├── cpp/                         # Game mechanics
└── python/                      # Python bindings

recipes/                        # Experiment definitions
├── prod/                        # Production-ready recipes
├── experiment/                  # Work-in-progress recipes
└── validation/                  # CI/stable test suites

observatory/                    # React training dashboard
gridworks/                      # Next.js web interface
```

---

## Core Concepts

### Recipes
Python modules defining tool functions that return configured Tool instances.

```python
# recipes/prod/my_recipe.py
def train() -> TrainTool:
    return TrainTool(training_env=..., evaluator=...)

def evaluate() -> EvaluateTool:
    return EvaluateTool(simulations=...)
```

**Usage**: `./tools/run.py train my_recipe run=exp_name`

### Tools
Pydantic config classes executing operations:
- `TrainTool`: Train agents
- `EvaluateTool`: Run policies on simulations
- `PlayTool`: Interactive testing
- `ReplayTool`: Visualize episodes
- `AnalysisTool`: Analyze results

### Policy URIs
Version-based addressing:
```
file://./train_dir/run_name/checkpoints/run_name:v5.mpt
s3://bucket/run_name/checkpoints/run_name:v10.mpt
```
Metadata embedded in filename (no database needed).

### Trainer
Main training loop with pluggable components:
- Vectorized environment for experience collection
- Loss functions (PPO, contrastive, etc.)
- Checkpointer for policy snapshots
- Evaluator for periodic validation
- StatsReporter for metrics logging

### Simulation
Evaluation runner:
1. Load policy from URI
2. Run episodes with fixed config
3. Collect statistics
4. Export to DuckDB

---

## Common Workflows

### Train
```bash
./tools/run.py train arena run=my_run trainer.total_timesteps=1000000
```

### Evaluate
```bash
./tools/run.py evaluate arena \
  policy_uris=file://./train_dir/my_run/checkpoints
```

### Interactive Play
```bash
./tools/run.py play arena policy_uri=file://./train_dir/my_run/checkpoints
```

### List Recipes
```bash
./tools/run.py --list
./tools/run.py train --list  # All recipes with train tool
```

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `/tools/run.py` | CLI entry point |
| `metta/rl/trainer.py` | Training loop |
| `metta/rl/checkpoint_manager.py` | Policy I/O |
| `metta/tools/train.py` | TrainTool |
| `metta/tools/eval.py` | EvaluateTool |
| `metta/sim/simulation.py` | Evaluation runner |
| `agent/policy.py` | Base Policy interface |
| `packages/mettagrid/` | C++ environment |
| `recipes/prod/` | Production recipes |
| `recipes/validation/` | CI/stable suites |

---

## Data Flow

### Training
```
Recipe (config)
  ↓
TrainTool.invoke()
  ↓
Trainer.train()
  ├─ VectorizedTrainingEnvironment (sample experience)
  ├─ Losses (compute gradients)
  ├─ Optimizer.step()
  └─ Checkpointer (save policies periodically)
  ↓
Checkpoints saved with URI format
```

### Evaluation
```
EvaluateTool.invoke()
  ↓
CheckpointManager.load_from_uri()
  ↓
Simulation.simulate()
  ├─ Run episodes with policy
  ├─ Collect statistics
  └─ Write to DuckDB
  ↓
Results: rewards, win rates, etc.
```

---

## Configuration

Uses **OmegaConf** for hierarchical configs:

```bash
# Overrides from CLI
./tools/run.py train arena \
  run=my_exp \
  trainer.total_timesteps=1000000 \
  trainer.batch_size=512 \
  policy_architecture=latent_attn_small
```

Key config objects:
- `TrainerConfig`: Learning params, loss weights, timesteps
- `TrainingEnvironmentConfig`: Parallel envs, curriculum
- `PolicyArchitecture`: Network structure (ViT, Fast, LSTM, Mamba)
- `SimulationConfig`: Evaluation scenario (suite, name, env)
- `EvaluatorConfig`: Eval simulations, frequency

---

## Policy Architectures

| Architecture | Characteristics |
|--------------|-----------------|
| **Fast** | CNN-based, highest SPS, lower expressivity |
| **ViT** | Vision Transformer, adaptable, three sizes (tiny/small/med) |
| **LSTM** | Recurrent, good for partial observability |
| **Mamba** | State-space models, balance of speed/expressivity |
| **Cortex** | Complex multi-component |

---

## Loss Functions

Available losses in `metta/rl/loss/`:
- **PPO**: Policy gradient with value baseline
- **GRPO**: Group reward optimization
- **Contrastive**: Self-supervised representation learning
- **Action Supervised**: Imitation learning
- **Dynamics**: World model learning
- **SL Kickstarter**: Supervised initialization

---

## Statistics & Analysis

**DuckDB** stores episode-level data:
- Episode rewards, lengths
- Agent actions, observations
- Mergeable across simulations

**Tools**:
- `EvalStatsDB`: Query eval results
- `analysis.py`: Generate scorecards
- Marimo/Jupyter notebooks: Interactive analysis

---

## Distributed Training

`DistributedHelper` handles:
- torch.distributed setup
- Multi-GPU batch scaling
- Gradient synchronization
- Checkpoint aggregation

Enable with: `trainer.num_gpus=4`

---

## Recipe Validation

**CI Suite** (`recipes/validation/ci_suite.py`):
- Quick smoke tests (10k timesteps, 5 min timeout)
- Run on every commit (planned)
- Verify no crashes

**Stable Suite** (`recipes/validation/stable_suite.py`):
- Full performance validation (100M-2B timesteps)
- Remote multi-GPU infrastructure
- Acceptance criteria (SPS, learning outcomes)

---

## Extension Points

### Add a Loss Function
1. Inherit from `Loss` in `metta/rl/loss/loss.py`
2. Implement `compute_loss(experience) -> scalar`
3. Register in `losses.py`
4. Use in recipe: `trainer.losses.my_loss.enabled=true`

### Add a Policy
1. Inherit from `Policy` in `agent/policy.py`
2. Implement `forward()`, `reset_memory()`, etc.
3. Create `PolicyArchitectureConfig` for configuration
4. Use in recipe: `policy_architecture=MyArchitectureConfig`

### Create a Recipe
1. Create `recipes/experiment/my_recipe.py` (or `recipes/prod/`)
2. Define: `def train() -> TrainTool`, `def evaluate() -> EvaluateTool`
3. Run: `./tools/run.py train my_recipe run=test`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `metta install core` |
| Build fails | `metta clean && metta configure` |
| Policy not found | Check URI format, verify file exists |
| Missing `__init__.py` | `uv run python devops/tools/ensure_recipe_packages.py` |
| Recipe not discovered | Ensure module is in `recipes/` with proper structure |

---

## Command Cheatsheet

```bash
# Discovery
./tools/run.py --list                    # All recipes
./tools/run.py train --list              # Recipes with train
./tools/run.py arena --list              # All tools in arena recipe

# Training
./tools/run.py train arena run=test      # Default config
./tools/run.py train arena \             # Custom config
  run=my_exp trainer.total_timesteps=1M

# Evaluation
./tools/run.py evaluate arena \
  policy_uris=file://./train_dir/my_run/checkpoints

# Interactive
./tools/run.py play arena                # Random policy play
./tools/run.py replay arena              # View replay

# Quality
metta ci                                 # All CI checks
metta pytest                             # Python tests
metta lint --fix metta/                  # Auto-fix linting
```

---

## Resources

- Full details: `/Users/mp/Metta-ai/metta/ARCHITECTURE.md`
- Project README: `/Users/mp/Metta-ai/metta/README.md`
- Development guide: `/Users/mp/Metta-ai/metta/CLAUDE.md`
- Tool runner docs: `metta/common/tool/README.md`
- MettaGrid docs: `packages/mettagrid/README.md`
- Policy guide: `agent/README.md` and `agent/src/metta/agent/policies/README.md`

