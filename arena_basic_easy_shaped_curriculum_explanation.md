# Arena Basic Easy Shaped Curriculum in Metta AI

The **Arena Basic Easy Shaped curriculum** is a simplified variant of the main arena curriculum, designed to provide an easier entry point for training while maintaining the core multi-agent dynamics and resource management challenges.

## What is the Arena Basic Easy Shaped Curriculum?

This curriculum is essentially a "gentler" version of the main arena curriculum, with simplified reward structures and easier resource conversion mechanics. It maintains the same core gameplay (multi-agent competition/cooperation, resource gathering, combat) but reduces complexity to accelerate learning.

## Key Differences from Main Arena:

### 1. **Simplified Reward Structure**
```python
# Main Arena vs Basic Easy Shaped rewards
arena_env.game.agent.rewards.inventory = {
    "heart": 1,      # Same
    "ore_red": 0.1,  # Lower than typical arena
    "battery_red": 0.8,  # Shaped for easier learning
    "laser": 0.5,   # Moderate combat rewards
    "armor": 0.5,   # Moderate defense rewards
    "blueprint": 0.5,  # Additional resource type
}
```

### 2. **Easier Resource Conversion**
```python
# Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
arena_env.game.objects["altar"].input_resources = {"battery_red": 1}
```

### 3. **Simplified Inventory Management**
```python
arena_env.game.agent.rewards.inventory_max = {
    "heart": 100,    # High heart capacity
    "ore_red": 1,    # Limited ore capacity
    "battery_red": 1,# Limited battery capacity
    "laser": 1,      # Limited combat items
    "armor": 1,      # Limited defense items
    "blueprint": 1,  # Limited special items
}
```

## Curriculum Learning System

The Basic Easy Shaped curriculum uses the same **bucketed task generation** as the main arena but with focused parameters:

### Reward Shaping Buckets:
- **ore_red**: [0, 0.1, 0.5, 0.9, 1.0] - Progressive ore value learning
- **battery_red**: [0, 0.1, 0.5, 0.9, 1.0] - Energy conversion learning
- **laser**: [0, 0.1, 0.5, 0.9, 1.0] - Combat strategy learning
- **armor**: [0, 0.1, 0.5, 0.9, 1.0] - Defense strategy learning

### Combat Dynamics:
- **Attack cost**: Variable laser consumption [1, 100] energy units
- **Attack effectiveness**: Maintains same damage mechanics as main arena
- **Defense mechanics**: Shield toggling with energy costs

## Training Dynamics

### Easier Learning Curve:
- **Simplified rewards**: Clearer feedback for basic behaviors
- **Easier conversion**: 1:1 battery to heart ratio (vs 3:1)
- **Limited inventory**: Forces efficient resource management
- **Progressive difficulty**: Same bucketed curriculum system

### Core Skills Developed:
- **Basic resource gathering**: Collecting ore_red from mines
- **Energy conversion**: Converting ore to batteries at generators
- **Goal achievement**: Converting batteries to hearts at altars
- **Combat fundamentals**: Basic attack/defense mechanics
- **Multi-agent awareness**: Cooperation and competition basics

## Use Cases

### 1. **Beginner Training**
- Starting point for new researchers
- Faster iteration during development
- Baseline performance testing

### 2. **Curriculum Research**
- Comparing learning efficiency
- Reward shaping studies
- Difficulty progression research

### 3. **Debugging and Testing**
- Isolating specific mechanics
- Controlled environment testing
- Performance baseline establishment

## Key Comparison with Main Arena

| Aspect             | Basic Easy Shaped  | Main Arena        |
| ------------------ | ------------------ | ----------------- |
| **Difficulty**     | Beginner-friendly  | Full complexity   |
| **Rewards**        | Simplified shaping | Complex dynamics  |
| **Conversion**     | 1:1 ratio          | 3:1 ratio         |
| **Inventory**      | Limited caps       | Variable limits   |
| **Combat**         | Same mechanics     | Same mechanics    |
| **Training Speed** | Faster convergence | Slower but deeper |

## Usage

```bash
# Train on Basic Easy Shaped curriculum
./tools/run.py experiments.recipes.arena_basic_easy_shaped.train --args run=my_easy_experiment

# Evaluate on basic arena tasks
./tools/run.py experiments.recipes.arena_basic_easy_shaped.evaluate --args policy_uri=wandb://run/my_easy_experiment
```

## Relation to Main Research

The Basic Easy Shaped curriculum serves as:
- **Onramp** for new researchers approaching Metta AI
- **Ablation study** for reward shaping effects
- **Development tool** for faster iteration
- **Teaching resource** for understanding core mechanics

While not the primary research curriculum, it provides valuable insights into learning efficiency and serves as a bridge between simple environments and the full complexity of the main arena curriculum.

## Related Files

- `metta/experiments/recipes/arena_basic_easy_shaped.py` - Implementation
- `metta/experiments/recipes/arena.py` - Main arena curriculum
- `metta/mettagrid/src/metta/mettagrid/config/envs.py` - Environment configs
