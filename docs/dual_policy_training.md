# Dual-Policy Training

The dual-policy training system allows you to train agents in environments where some agents are controlled by the main policy being trained, while others are controlled by NPC policies (either scripted behaviors or checkpoint-based policies). This enables studying how agents learn in mixed populations and can be useful for curriculum learning, adversarial training, and multi-agent research.

## Overview

The dual-policy system splits the agent population into two groups:
- **Policy A**: The main policy being trained (configurable percentage)
- **NPC Policy**: Either a scripted behavior or a checkpoint-based policy

Both policies receive rewards and their performance is tracked separately, allowing you to compare how different agent types perform in the same environment.

## Configuration

Enable dual-policy training by adding the `dual_policy` configuration to your trainer config:

```yaml
dual_policy:
  enabled: true
  policy_a_percentage: 0.5  # 50% of agents use the main policy

  # Choose NPC type
  npc_type: "scripted"  # or "checkpoint"

  # For scripted NPCs
  scripted_npc:
    type: "roomba"  # or "grid_search"
    approach_items: true
    interact_with_items: true
    roomba_direction: "clockwise"  # or "counterclockwise"
    grid_search_pattern: "spiral"  # or "snake", "random"

  # For checkpoint-based NPCs
  npc_policy_uri: "path/to/npc/policy/checkpoint"  # required for checkpoint type
```

## Scripted NPC Types

### Roomba Behavior
- Moves in a consistent direction
- Turns when hitting walls (configurable direction)
- Can approach and interact with items in field of view
- Good for simple exploration patterns

### Grid Search Behavior
- Systematic exploration patterns
- Options: spiral, snake, random
- Can approach and interact with items
- More structured exploration than roomba

## Checkpoint-Based NPCs

Use a previously trained policy as the NPC:
- Load from local checkpoint path
- Load from WandB artifact URI
- Supports any policy compatible with the environment

## Metrics and Logging

The system tracks and logs the following metrics to WandB:

- `dual_policy/policy_a_reward`: Average reward for main policy agents
- `dual_policy/policy_a_reward_total`: Total reward for main policy agents
- `dual_policy/npc_reward`: Average reward for NPC agents
- `dual_policy/npc_reward_total`: Total reward for NPC agents
- `dual_policy/combined_reward`: Average combined reward
- `dual_policy/combined_reward_total`: Total combined reward
- `dual_policy/policy_a_agent_count`: Number of main policy agents
- `dual_policy/npc_agent_count`: Number of NPC agents

## Example Usage

### Basic Scripted NPC Training

```bash
# Train with roomba NPCs
python tools/train.py trainer=dual_policy_example

# Train with grid search NPCs
python tools/train.py trainer=dual_policy_example dual_policy.scripted_npc.type=grid_search
```

### Checkpoint-Based NPC Training

```bash
# Train with a specific checkpoint as NPC
python tools/train.py \
  trainer=dual_policy_example \
  dual_policy.npc_type=checkpoint \
  dual_policy.npc_policy_uri=path/to/checkpoint
```

### Custom Configuration

```bash
# Train with 70% main policy, 30% NPC
python tools/train.py \
  trainer=dual_policy_example \
  dual_policy.policy_a_percentage=0.7
```

## Implementation Details

### Agent Assignment
- Agents are assigned to policies at the start of each episode
- Assignment is fixed throughout the episode
- Each environment contains both policy types in the configured proportion

### Reward Tracking
- Rewards are tracked separately for each policy type
- Statistics are aggregated across episodes
- Both individual and cumulative rewards are logged

### Experience Collection
- Only the main policy's experience is used for training
- NPC experience is tracked for statistics but not used for learning
- This ensures the main policy learns from its own interactions

## Use Cases

### Curriculum Learning
- Start with simple scripted NPCs
- Gradually increase complexity
- Use checkpoint-based NPCs from previous training runs

### Adversarial Training
- Train against increasingly difficult opponents
- Use checkpoint-based NPCs from different training stages
- Study robustness and adaptation

### Multi-Agent Research
- Study emergent behaviors in mixed populations
- Compare different agent types
- Analyze cooperation and competition

### Debugging and Analysis
- Use scripted NPCs for controlled experiments
- Compare against baseline behaviors
- Analyze learning dynamics

## Limitations

- Scripted NPCs are simplified implementations
- LSTM state handling for checkpoint NPCs is basic
- Experience collection only uses main policy data
- Agent assignments are fixed per episode

## Future Enhancements

- More sophisticated scripted behaviors
- Dynamic agent reassignment
- Experience sharing between policies
- Multi-policy training (more than 2 policies)
- Adaptive NPC difficulty
