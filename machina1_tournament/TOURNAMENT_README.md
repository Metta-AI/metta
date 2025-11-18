# Policy Tournament System

A tournament system for evaluating policies using the **value-of-replacement** metric.

## Concept

**Value of Replacement (VOR)** measures how much better or worse games are when a specific policy participates, compared to the average:

```
VOR(policy) = Mean(hearts in games where policy played) - Mean(hearts in all games)
```

- **Positive VOR**: Games with this policy score better than average → Good policy
- **Negative VOR**: Games with this policy score worse than average → Bad policy
- **Zero VOR**: Policy performs at the average level

## Quick Start

### 1. Basic Tournament (Default Pool)

Run a tournament with auto-generated random and scripted agents:

```bash
cd /Users/bullm/Documents/metta2

# Quick test with 20 episodes
uv run python metta/machina1_tournament/tournament.py --num-episodes 20 --pool-size 8

# Full tournament with 100 episodes and 16 policies
uv run python metta/machina1_tournament/tournament.py --num-episodes 100 --pool-size 16

# Save results to file
uv run python metta/machina1_tournament/tournament.py --num-episodes 100 -o tournament_results.json
```

### 2. Custom Policy Pool

Create a YAML file defining your policies (see `example_policy_pool.yaml`):

```yaml
policies:
  - name: MyPolicy_1
    class_path: metta.agent.policies.fast.FastPolicy
    data_path: ./checkpoints/policy1.pt
  
  - name: MyPolicy_2
    class_path: metta.agent.policies.fast.FastPolicy
    data_path: ./checkpoints/policy2.pt
  
  - name: Random_1
    class_path: mettagrid.policy.random.RandomPolicy
  # ... more policies
```

Then run:

```bash
uv run python metta/machina1_tournament/tournament.py \
    --policies my_policy_pool.yaml \
    --num-episodes 200 \
    -o my_results.json
```

### 3. Different Missions

Test on different mission configurations:

```bash
# Machina 1 (4 agents)
uv run python metta/machina1_tournament/tournament.py -m machina_1 --team-size 4 -n 100

# Training Facility (can adjust team size)
uv run python metta/machina1_tournament/tournament.py -m training_facility_1 --team-size 4 -n 100
```

## Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--mission` | `-m` | `machina_1` | Mission to play |
| `--num-episodes` | `-n` | `100` | Number of tournament games |
| `--team-size` | `-t` | `4` | Agents per team |
| `--pool-size` | `-p` | `16` | Size of policy pool (when auto-generating) |
| `--policies` | | `None` | Path to policy pool YAML file |
| `--seed` | `-s` | `42` | Random seed |
| `--render` | `-r` | `False` | Render games (very slow) |
| `--output` | `-o` | `None` | Output JSON file for results |
| `--max-steps` | | `1000` | Max steps per episode |

## Example Output

```
Tournament Results
==================

Overall Statistics
  Total Episodes: 100
  Total Hearts: 450.0
  Mean Hearts/Game: 4.50
  Std Hearts/Game: 2.31

Policy Rankings (by Value of Replacement)
┌──────┬─────────────────┬───────┬─────────────┬──────────────────────┬───────────┐
│ Rank │ Policy Name     │ Games │ Mean Hearts │ Value of Replacement │ Positions │
├──────┼─────────────────┼───────┼─────────────┼──────────────────────┼───────────┤
│    1 │ Unclipping_1    │    25 │        6.20 │               +1.70  │ 7,6,5,7   │
│    2 │ Baseline_3      │    23 │        5.43 │               +0.93  │ 6,5,6,6   │
│    3 │ Baseline_1      │    26 │        4.85 │               +0.35  │ 7,6,7,6   │
│    4 │ Random_5        │    24 │        4.67 │               +0.17  │ 6,6,6,6   │
│    5 │ Random_1        │    27 │        4.52 │               +0.02  │ 7,7,6,7   │
│   ...│ ...             │   ... │         ... │                  ... │       ... │
│   14 │ Random_3        │    23 │        3.22 │               -1.28  │ 6,5,6,6   │
│   15 │ Random_7        │    26 │        2.85 │               -1.65  │ 7,6,6,7   │
│   16 │ Baseline_2      │    25 │        2.40 │               -2.10  │ 6,7,6,6   │
└──────┴─────────────────┴───────┴─────────────┴──────────────────────┴───────────┘
```

## Output File Format

When using `--output`, the JSON file contains:

```json
{
  "tournament_summary": {
    "num_episodes": 100,
    "overall_mean_hearts": 4.50,
    "overall_std_hearts": 2.31
  },
  "policy_rankings": [
    {
      "rank": 1,
      "policy_idx": 8,
      "policy_name": "Unclipping_1",
      "games_played": 25,
      "mean_hearts_when_playing": 6.20,
      "value_of_replacement": 1.70
    }
    // ... more policies
  ],
  "game_results": [
    {
      "episode_id": 0,
      "policy_indices": [3, 7, 12, 15],
      "total_hearts": 5.0,
      "steps": 347
    }
    // ... more games
  ]
}
```

## Use Cases

1. **Model Selection**: Compare multiple trained checkpoints to find the best performer
2. **Architecture Comparison**: Test different policy architectures against each other
3. **Baseline Evaluation**: See how your trained policies compare to scripted baselines
4. **Ensemble Selection**: Identify which policies work well together
5. **Training Progress**: Track VOR over time as you train new checkpoints

## Statistical Considerations

- **Sample Size**: Use at least 50-100 episodes for stable VOR estimates
- **Team Sampling**: Policies are sampled uniformly without replacement each game
- **Position Effects**: The "Positions" column shows how often a policy played in each agent position (0,1,2,3)
- **Variance**: High-variance policies may have unstable VOR; check std in results

## Advanced Usage

### Tracking Training Progress

```bash
# After each training checkpoint, run tournament
for checkpoint in checkpoints/*.pt; do
    # Create temp policy pool with this checkpoint
    python create_pool.py --checkpoint $checkpoint -o temp_pool.yaml
    python metta/machina1_tournament/tournament.py --policies temp_pool.yaml -o results_$checkpoint.json
done

# Analyze VOR over time
python analyze_vor_progression.py results_*.json
```

### Team Composition Analysis

Modify the tournament script to test specific team compositions (e.g., 3 trained + 1 random) instead of random sampling.

## Files

- `tournament.py` - Main tournament script
- `example_policy_pool.yaml` - Example policy pool configuration
- `multi_policy_example.py` - Full CLI example for running 4 policies
- `simple_multi_policy.py` - Simple example showing core mechanics
- `TOURNAMENT_README.md` - This file

