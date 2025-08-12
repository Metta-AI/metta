## Sequence Learning (ColorTree)

This document outlines how to train, evaluate, and analyze sequence-learning experiments using the ColorTree environment.

### Concepts

- **Environment**: `env/mettagrid/colortree_easy` — 2 colors, 4-slot target sequence, 128 agents, 64 steps. Reward modes: `precise` (all-or-nothing), `partial`, `dense`.
- **Fixed vs Random**:
  - Fixed: Specify a single target sequence via overrides.
  - Random: Use the `ColorTreeRandomFromSetCurriculum` to vary the sequence each episode.

### Training

You can launch training locally or via your cluster wrapper. Below are plain Hydra command lines.

Fixed single sequence (example: 1010):

```bash
./run.py \
  +trainer.env=env/mettagrid/colortree_easy \
  sim=colortree \
  agent=fast \
  trainer.bptt_horizon=64 \
  "+trainer.env_overrides.game.actions.color_tree.target_sequence=[1,0,1,0]" \
  "+trainer.env_overrides.game.actions.color_tree.trial_sequences=[[1,0,1,0]]" \
  +trainer.env_overrides.game.actions.color_tree.num_trials=1 \
  "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise"
```

Random sequence curriculum (2 colors, length 4):

```bash
./run.py \
  trainer.curriculum=/env/mettagrid/curriculum/colortree_easy_random \
  sim=colortree \
  agent=fast \
  trainer.bptt_horizon=64 \
  "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise"
```

Recurrent agent variant (swap agent for an LSTM/Transformer recurrent policy):

```bash
./run.py \
  +trainer.env=env/mettagrid/colortree_easy \
  sim=colortree \
  agent._target_=metta.agent.external.lstm_transformer.Recurrent \
  trainer.bptt_horizon=64 \
  "+trainer.env_overrides.game.actions.color_tree.target_sequence=[1,0,1,0]" \
  "+trainer.env_overrides.game.actions.color_tree.trial_sequences=[[1,0,1,0]]" \
  +trainer.env_overrides.game.actions.color_tree.num_trials=1 \
  "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise"
```

Notes:
- Use `bptt_horizon` suited to your architecture (e.g., 16, 64).
- For fixed-sequence runs, ensure `trial_sequences` length matches the `target_sequence` length and set `num_trials=1` so the base target applies consistently.

### Evaluation (single env)

Run an evaluation suite programmatically using `tools.sim`. The simplest path is to provide a single env and episode along with policy artifact URI.

```bash
python -m tools.sim \
  sim=sim_suite \
  +sim.name=colortree_eval \
  +sim.simulations.task.env=env/mettagrid/colortree_easy \
  sim.num_episodes=1 \
  device=cpu \
  vectorization=serial \
  run=<your_eval_run_name> \
  policy_uri=wandb://<entity>/<project>/model/<artifact_base>:v7
```

Override the target sequence at eval time (e.g., force 1010):

```bash
python -m tools.sim \
  sim=sim_suite \
  +sim.name=colortree_eval \
  +sim.simulations.task.env=env/mettagrid/colortree_easy \
  sim.num_episodes=1 \
  device=cpu \
  vectorization=serial \
  run=<your_eval_run_name> \
  policy_uri=wandb://<entity>/<project>/model/<artifact_base>:v7 \
  +sim.simulations.task.env_overrides.game.actions.color_tree.num_trials=1 \
  "+sim.simulations.task.env_overrides.game.actions.color_tree.target_sequence=[1,0,1,0]"
```

### Change number of colors and sequence length

You can change both the number of colors and the sequence length via overrides. Two common cases:

#### Fixed, 2 colors, 2-length sequence (example: 10)

```bash
./run.py \
  +trainer.env=env/mettagrid/colortree_easy \
  sim=colortree \
  agent=fast \
  trainer.bptt_horizon=64 \
  "+trainer.env_overrides.game.actions.color_tree.color_to_item={0: ore_red, 1: ore_green}" \
  "+trainer.env_overrides.game.actions.color_tree.target_sequence=[1,0]" \
  "+trainer.env_overrides.game.actions.color_tree.trial_sequences=[[1,0]]" \
  +trainer.env_overrides.game.actions.color_tree.num_trials=1 \
  "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise"
```

Evaluate the same 2-length sequence:

```bash
python -m tools.sim \
  sim=sim_suite \
  +sim.name=colortree_eval_len2 \
  +sim.simulations.task.env=env/mettagrid/colortree_easy \
  sim.num_episodes=1 \
  device=cpu \
  vectorization=serial \
  run=<your_eval_run_name> \
  policy_uri=wandb://<entity>/<project>/model/<artifact_base>:v7 \
  +sim.simulations.task.env_overrides.game.actions.color_tree.num_trials=1 \
  "+sim.simulations.task.env_overrides.game.actions.color_tree.target_sequence=[1,0]"
```

#### Random curriculum with 2-length sequences

Use the random curriculum config and override `sequence_length=2` (and optionally `num_colors`). It will auto-detect colors from the env if not provided.

```bash
./run.py \
  trainer.curriculum=/env/mettagrid/curriculum/colortree_easy_binary \
  +trainer.curriculum.sequence_length=2 \
  +trainer.curriculum.num_colors=2 \
  sim=colortree \
  agent=fast \
  trainer.bptt_horizon=64 \
  "+trainer.env_overrides.game.actions.color_tree.color_to_item={0: ore_red, 1: ore_green}" \
  "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise"
```

For 3 colors and 2-length sequences, update both mapping and `num_colors`:

```bash
./run.py \
  trainer.curriculum=/env/mettagrid/curriculum/colortree_easy_random \
  +trainer.curriculum.sequence_length=2 \
  +trainer.curriculum.num_colors=3 \
  sim=colortree \
  agent=fast \
  trainer.bptt_horizon=64 \
  "+trainer.env_overrides.game.actions.color_tree.color_to_item={0: ore_red, 1: ore_green, 2: ore_blue}" \
  "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise"
```

#### Evaluating all 4 sequences for length-2 (00, 01, 10, 11)

The provided `tools/ct_sequence_heatmap.py` targets length-4. For length-2, you can quickly sweep the 4 sequences with a shell loop:

```bash
for seq in "[0,0]" "[0,1]" "[1,0]" "[1,1]"; do
  python -m tools.sim \
    sim=sim_suite \
    +sim.name=ct_len2_eval \
    +sim.simulations.task.env=env/mettagrid/colortree_easy \
    sim.num_episodes=1 \
    device=cpu \
    vectorization=serial \
    run="ct_len2_${seq}" \
    policy_uri=wandb://<entity>/<project>/model/<artifact_base>:v7 \
    +sim.simulations.task.env_overrides.game.actions.color_tree.num_trials=1 \
    "+sim.simulations.task.env_overrides.game.actions.color_tree.target_sequence=${seq}"
done
```

Capture scores from each run and build a small 2×2 table or heatmap with your plotting tool of choice.

### Seed sweep summary (versions × seeds)

Use `tools/ct_seed_sweep.py` to evaluate a set of versions and seeds and print per-version averages.

```bash
python /workspace/metta/tools/ct_seed_sweep.py \
  --run-name-base <artifact_base> \
  --versions v7 v8 v9 v10 \
  --seeds 1 2 3 4 5 \
  --env env/mettagrid/colortree_easy \
  --episodes 1 \
  --device cpu \
  --vectorization serial \
  --out ./train_dir/ct_seed_sweep/results.csv
```

### Per-sequence heatmap (16 sequences)

Use `tools/ct_sequence_heatmap.py` to evaluate all 16 two-color sequences (length 4) and generate per-version 4×4 heatmaps.

Quick single-version run (faster to get a read):

```bash
python /workspace/metta/tools/ct_sequence_heatmap.py \
  --run-name-base <artifact_base> \
  --versions v7 \
  --seeds 1 2 3 4 5 \
  --env env/mettagrid/colortree_easy \
  --episodes 1 \
  --device cpu \
  --vectorization serial \
  --out-dir /workspace/metta/train_dir/ct_sequence_heatmap_v7
```

Outputs:
- `results.csv`: raw rows (version, seed, sequence, score)
- `results_aggregated.csv`: averages per (version, sequence)
- `heatmap_<version>.png`: 4×4 matrix (rows = first two bits, cols = last two bits)

Speed tips:
- Use GPU if available: `device=cuda`.
- Temporarily reduce steps: `+sim.simulations.task.env_overrides.game.max_steps=32`.
- Temporarily reduce agents: `+sim.simulations.task.env_overrides.game.num_agents=16`.
- For large grids of runs (many versions × seeds), consider batching multiple sequences into one process to amortize artifact loading.

### Random curriculum details

`/env/mettagrid/curriculum/colortree_easy_random.yaml` uses `ColorTreeRandomFromSetCurriculum`, which programmatically generates a pool of sequences (including alternating, palindrome-like, and sampled patterns) and randomly selects one each episode. For precise reproduction or controlled evaluation, override `target_sequence`, `trial_sequences`, and `num_trials` at eval time as shown above.

### Ternary (3-color) with 2-length sequences and BPTT sweep

Recipes:
- `recipes/colortree_ternary_random_bptt_sweep.sh` (random sequences)
- `recipes/colortree_ternary_fixed_bptt_sweep.sh` (fixed sequences like [2,0], [1,2])

Random sweep runs with:
- Random curriculum: `/env/mettagrid/curriculum/colortree_easy_random`
- `+trainer.curriculum.sequence_length=2`
- `+trainer.curriculum.num_colors=3`
- `trainer.bptt_horizon` in `[16, 32, 64, 128]`
- Run names include the bptt value and seed for easy tracking

Example call:

```bash
./recipes/colortree_ternary_random_bptt_sweep.sh
# or with fixed seed and extra overrides
./recipes/colortree_ternary_random_bptt_sweep.sh seed=123 +trainer.env_overrides.game.num_agents=64
```

Fixed sequence sweep (e.g., [2,0] and [1,2]) runs:

```bash
./recipes/colortree_ternary_fixed_bptt_sweep.sh
# or with fixed seed and overrides
./recipes/colortree_ternary_fixed_bptt_sweep.sh seed=123 +trainer.env_overrides.game.num_agents=64
```

Binary (2-color) mixed comparison suite:

- `recipes/colortree_binary_mixed_suite.sh` launches a mixed set:
  - Random binary (2-color) with `+trainer.curriculum.sequence_length=2`, `+trainer.curriculum.num_colors=2`
  - Fixed binary sequences [1,0] and [0,1]
  - All with embedded bptt and seed in run names


