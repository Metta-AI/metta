# Arena Curriculum in Metta AI

Based on the Metta AI project documentation and code, the **arena curriculum** is a training curriculum used to train AI agents in a complex multi-agent gridworld environment. Here's what it entails:

## What is the Arena Curriculum?

The arena curriculum is one of the primary training environments in Metta AI, designed to study the emergence of cooperation and alignment in multi-agent AI systems. It's a sophisticated gridworld environment where agents learn through a **curriculum learning** approach that progressively increases task difficulty.

## Key Components:

### 1. **Arena Environment**
- **Grid-based world**: 25x25 grid with various objects and resources
- **Multiple agents**: Typically 24 agents competing and cooperating
- **Resources and buildings**:
  - **Mines** (mine_red): Generate ore_red resources
  - **Generators** (generator_red): Convert resources to energy (battery_red)
  - **Altars**: Agents convert energy to "heart" rewards
  - **Lasers** and **Armor**: Combat equipment
- **Energy management**: All actions cost energy, requiring efficient resource management

### 2. **Curriculum Learning System**
The arena curriculum uses a **bucketed task generation** system that creates variations of the base arena environment by randomly sampling different parameters:

```python
# Example bucket configurations from the code:
arena_tasks.add_bucket("game.agent.rewards.inventory.ore_red", [0, 0.1, 0.5, 0.9, 1.0])
arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])  # Enable/disable combat
arena_tasks.add_bucket("game.objects.mine_red.initial_resource_count", [0, 1])  # Randomize starting resources
```

### 3. **Training Dynamics**
- **Cooperation vs Competition**: Agents can share resources, communicate via markers, and form alliances
- **Combat system**: Agents can attack others but must manage energy costs
- **Progressive difficulty**: The curriculum automatically generates new task variations to prevent agents from reaching optimal strategies too quickly
- **Evaluation suites**: Includes both basic cooperation tasks and combat scenarios

### 4. **How to Use It**
You can train agents on the arena curriculum using:

```bash
# Train on the arena curriculum
./tools/run.py experiments.recipes.arena.train --args run=my_experiment

# Evaluate trained agents
./tools/run.py experiments.recipes.arena.evaluate --args policy_uri=wandb://run/my_experiment
```

## Purpose

The arena curriculum serves as a **model organism** for studying how social dynamics (kinship, cooperation, competition) affect the development of general intelligence in AI agents. By creating an environment where incremental intelligence is rewarded, it aims to foster the emergence of generally intelligent agents capable of complex social behaviors.

The curriculum design ensures that agents face an ever-evolving challenge, preventing them from reaching a Nash equilibrium and encouraging continuous learning and adaptation.

## Additional Context

From the Metta AI project's README and codebase:

- The arena is part of a reinforcement learning framework focused on multi-agent cooperation
- Agents learn policies that enhance their fitness through environmental interaction
- The system includes kinship structures simulating relationships from close kin to strangers
- Training involves both competitive and cooperative dynamics to smooth the behavioral space

## Related Files in the Project

- `metta/experiments/recipes/arena.py` - Main arena curriculum implementation
- `metta/mettagrid/src/metta/mettagrid/config/envs.py` - Arena environment configuration
- `metta/metta/cogworks/curriculum/` - Curriculum learning system
- `metta/metta/cogworks/curriculum/curriculum.py` - Core curriculum classes
