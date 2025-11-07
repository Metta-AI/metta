# CoGs vs Clips Training Recipes

Training recipes for the Cogs vs Clips eval missions, supporting curriculum learning across multiple missions and
difficulty variants. All `experiments.recipes.cvc.*` entry points now proxy directly to
`experiments.recipes.cogs_v_clips`, so there is a single source of truth for missions, variants, and trainer defaults.
The commands are still provided for continuity, but any options passed to them are forwarded unchanged to the shared
recipe implementation.

## Quick Start

### Batch Experiments (Skypilot)

Launch multiple training runs on cloud infrastructure:

```bash
# Test with debug config (recommended first!)
uv run python experiments/recipes/cvc/experiment.py debug_single

# Run specific experiments
uv run python experiments/recipes/cvc/experiment.py small_4cogs medium_4cogs

# Run all standard experiments (excludes debug configs)
uv run python experiments/recipes/cvc/experiment.py

# Programmatic usage
uv run python -c "from experiments.recipes.cvc.experiment import experiment; experiment(configs=['debug_single'])"
```

**Available experiment configs:**

| Config               | Agents | GPUs | Steps | Description                   |
| -------------------- | ------ | ---- | ----- | ----------------------------- |
| `debug_single`       | 2      | 1    | 5M    | Quick test on single mission  |
| `small_1cog`         | 1      | 2    | 20M   | Small maps, single agent      |
| `small_2cogs`        | 2      | 2    | 20M   | Small maps, 2 agents          |
| `small_4cogs`        | 4      | 4    | 30M   | Small maps, 4 agents          |
| `medium_4cogs`       | 4      | 4    | 40M   | Medium maps, 4 agents         |
| `coordination_4cogs` | 4      | 4    | 40M   | Coordination missions         |
| `full_1cog`          | 1      | 4    | 50M   | Full curriculum, single agent |
| `full_4cogs`         | 4      | 8    | 100M  | Full curriculum, 4 agents     |
| `full_8cogs`         | 8      | 8    | 100M  | Full curriculum, 8 agents     |

**Recommended workflow:**

1. Start with `debug_single` to test the pipeline (~1 hour)
2. Run `small_4cogs` for initial results (~6 hours)
3. Run `full_4cogs` for publication-quality results (~24 hours)

### Basic Training

Train locally on small maps with 4 agents:

```bash
# Train with default curriculum (small/medium missions, 4 agents)
uv run ./tools/run.py experiments.recipes.cvc.curriculum.train run=cvc_default

# Train on single mission (fast, for debugging)
uv run ./tools/run.py experiments.recipes.cvc.single_mission.train run=cvc_single

# Train on small maps only
uv run ./tools/run.py experiments.recipes.cvc.small_maps.train run=cvc_small

# Train on coordination-heavy missions
uv run ./tools/run.py experiments.recipes.cvc.coordination.train run=cvc_coord
```

### Evaluation

```bash
# Evaluate a trained policy on all eval missions
uv run ./tools/run.py experiments.recipes.cvc.evaluation.evaluate \\
    policy_uris=file://./checkpoints/cvc_default/latest

# Evaluate on specific missions
uv run ./tools/run.py experiments.recipes.cvc.evaluation.evaluate \\
    policy_uris=file://./checkpoints/cvc_default/latest \\
    subset='["extractor_hub_30", "oxygen_bottleneck"]'

# Evaluate on hard difficulty
uv run ./tools/run.py experiments.recipes.cvc.evaluation.evaluate \\
    policy_uris=file://./checkpoints/cvc_default/latest \\
    difficulty=hard
```

### Interactive Play

```bash
# Play with a trained policy
uv run ./tools/run.py experiments.recipes.cvc.small_maps.play \\
    policy_uri=file://./checkpoints/cvc_default/latest \\
    mission_name=extractor_hub_30 \\
    num_cogs=4

# Play without a policy (random actions)
uv run ./tools/run.py experiments.recipes.cvc.curriculum.play
```

### Built-in Variant Presets

For every CVC entry point you can call:

- `train_easy` / `train_shaped`
- `play_easy` / `play_shaped`

Examples:

```bash
uv run ./tools/run.py experiments.recipes.cvc.small_maps.train_easy run=my_easy_run
uv run ./tools/run.py experiments.recipes.cvc.medium_maps.play_shaped mission_name=collect_resources_spread
uv run ./tools/run.py experiments.recipes.cvc.curriculum.train_shaped run=my_curriculum
uv run ./tools/run.py experiments.recipes.cvc.single_mission.train_easy mission_name=training_facility.harvest
```

Each helper defaults to the variant bundles below, but you can still override `variants=` or `eval_variants=` if needed.

### Variant Presets (manual override)

- **Easy starter:** `lonely_heart`, `pack_rat`, `neutral_faced`

  ```bash
  uv run ./tools/run.py experiments.recipes.cvc.small_maps.train \\
      run=my_easy_run \\
      variants='["lonely_heart","pack_rat","neutral_faced"]'
  ```

- **Reward-shaped:** `lonely_heart`, `heart_chorus`, `pack_rat`, `neutral_faced`, `extractor_base`

  ```bash
  uv run ./tools/run.py experiments.recipes.cvc.small_maps.train \\
      run=my_shaped_run \\
      variants='["lonely_heart","heart_chorus","pack_rat","neutral_faced","extractor_base"]'

  > Note: `extractor_base` only affects BaseHub/Machina maps (e.g., training_facility). It is skipped automatically on fixed eval layouts.
  ```

Mix and match with difficulty variants (for example, add `difficulty=hard` or append `"clipped_oxygen"` to the list) as
needed—the recipes forward everything to the `cogames` variant parser.

## Recipe Functions

### Training Functions

- **`train(num_cogs, curriculum, base_missions)`** - Main training with curriculum
  - Creates curriculum across multiple missions and difficulty parameters
  - Default: 4 agents, 6 diverse missions
  - Evaluates on full eval suite

- **`train_single_mission(mission_name, num_cogs)`** - Train on single mission
  - No curriculum variation
  - Fast for debugging
  - Still evaluates on full suite

- **`train_small_maps(num_cogs)`** - Train on 30x30 maps
  - Missions: extractor_hub_30, collect_resources_classic, oxygen_bottleneck
  - Good for quick iterations

- **`train_medium_maps(num_cogs)`** - Train on 50x50 maps
  - Missions: extractor_hub_50, collect_resources_spread, energy_starved
  - Balanced training

- **`train_large_maps(num_cogs)`** - Train on 70x70+ maps
  - Missions: extractor_hub_70, collect_far, divide_and_conquer
  - Requires more agents (default: 8)

- **`train_coordination(num_cogs)`** - Multi-agent coordination focus
  - Missions: go_together, divide_and_conquer, collect_resources_spread
  - Emphasizes cooperation

### Evaluation Functions

- **`evaluate(policy_uris, num_cogs, difficulty, subset)`** - Evaluate on missions
  - Tests policy on eval suite
  - Can filter by difficulty or mission subset

- **`make_eval_suite(num_cogs, difficulty, subset)`** - Create evaluation suite
  - Returns list of SimulationConfig objects
  - Used internally by evaluate()

### Play Functions

- **`play(policy_uri, mission_name, num_cogs)`** - Play a specific mission
  - Interactive visualization
  - Useful for debugging trained policies

- **`play(policy_uri, mission_name, num_cogs)`** _(via `experiments.recipes.cvc.curriculum.play`)_ - Play default
  training env
  - Defaults to extractor_hub_30

### Utility Functions

- **`make_training_env(num_cogs, mission_name)`** - Create single env config
  - Returns MettaGridConfig
  - Useful for custom training setups

- **`make_curriculum(num_cogs, base_missions, ...)`** - Create custom curriculum
  - Varies mission types, episode length, and reward weights
  - Each mission has built-in difficulty tuning (efficiency, max_uses, energy regen)
  - Uses Learning Progress algorithm by default

## Available Missions

### Small Maps (30x30)

- `extractor_hub_30` - Basic hub layout
- `oxygen_bottleneck` - Oxygen paces assembly
- `collect_resources_classic` - Classic balanced layout

### Medium Maps (50x50)

- `extractor_hub_50` - Medium hub layout
- `collect_resources_spread` - Resources scattered nearby
- `energy_starved` - Low energy regeneration

### Large Maps (70x70+)

- `extractor_hub_70` - Large hub
- `extractor_hub_80` - Extra large hub
- `extractor_hub_100` - Huge hub (use with 8+ agents)
- `collect_far` - Resources far from base
- `divide_and_conquer` - Regionalized resources

### Coordination-Heavy

- `go_together` - Favors collective glyphing (min 2 agents)
- `single_use_swarm` - Single-use stations, team must coordinate (min 2 agents)

## Curriculum Parameters

The curriculum varies:

1. **Mission Type**
   - Different maps, layouts, and resource distributions
   - Each mission has unique difficulty characteristics (efficiency, max_uses, energy_regen)

2. **Episode Length** (750-1500 steps)
   - Longer episodes = more time to complete complex strategies
   - Shorter episodes = faster feedback, encourages efficiency

3. **Reward Weights** (0.1-1.0 per heart)
   - Affects learning signal strength
   - Lower weights = more exploration, higher weights = more exploitation

The missions themselves provide variation in:

- **Station Efficiency**: Different conversion rates for extractors and chargers
- **Max Uses**: Some missions have limited extractor uses
- **Energy Regeneration**: Varies from 1-2 per step depending on mission
- **Spatial Layout**: Distance to resources, bottlenecks, open vs constrained

## Comparison to Scripted Agents

The scripted agents achieved:

- **Baseline**: 41.5% success (151/364 tests), best with 4 agents (54.9%)
- **UnclippingAgent**: 40.5% success (274/676 tests), best with 4 agents (45.0%)

## Example Workflow

```bash
# 1. Train on small maps for quick iteration
uv run ./tools/run.py experiments.recipes.cvc.small_maps.train \\
    run=cvc_small \\
    trainer.total_timesteps=10_000_000

# 2. Watch agent performance during training
# (Check wandb or tensorboard for metrics)

# 3. Evaluate on full suite
uv run ./tools/run.py experiments.recipes.cvc.evaluation.evaluate \\
    policy_uris=file://./checkpoints/cvc_small/latest

# 4. Play to visualize behavior
uv run ./tools/run.py experiments.recipes.cvc.small_maps.play \\
    policy_uri=file://./checkpoints/cvc_small/latest \\
    mission_name=extractor_hub_30 \\
    num_cogs=4

# 5. Scale up to full curriculum
uv run ./tools/run.py experiments.recipes.cvc.curriculum.train \\
    run=cvc_full \\
    trainer.total_timesteps=50_000_000 \\
    num_cogs=4
```

## Notes

- The recipe evaluates on **standard difficulty** by default
- Multi-agent missions (go_together, single_use_swarm) require ≥2 agents
- Large maps (80x80, 100x100) work best with 8 agents
- Curriculum uses **Learning Progress** algorithm for adaptive task selection
- All eval missions use simplified heart recipes for faster assembly

## Related Files

- **Mission Definitions**: `packages/cogames/src/cogames/cogs_vs_clips/evals/eval_missions.py`
- **Difficulty Variants**: `packages/cogames/src/cogames/cogs_vs_clips/evals/difficulty_variants.py`
- **Scripted Agent Baselines**: `packages/cogames/src/cogames/policy/scripted_agent/`
- **Evaluation Results**: `experiments/SCRIPTED_AGENT_EVALUATION.md`
