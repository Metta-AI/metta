# Metta AI Curricula Overview

This document provides an overview of all curricula and evaluation suites available in the Metta AI project.

## Main Training Curricula

### 1. **Arena Curriculum** (`arena_curriculum_explanation.md`)
- **Focus**: Multi-agent cooperation and competition
- **Agents**: 24 agents per environment
- **Environment**: Complex resource management with combat
- **Skills**: Social dynamics, resource optimization, strategic combat
- **Usage**: `./tools/run.py experiments.recipes.arena.train`

### 2. **Navigation Curriculum** (`navigation_curriculum_explanation.md`)
- **Focus**: Spatial navigation and pathfinding
- **Agents**: 4 agents per environment
- **Environment**: Mazes and varied terrain
- **Skills**: Path optimization, exploration strategies, spatial memory
- **Usage**: `./tools/run.py experiments.recipes.navigation.train`

## Specialized Curricula

### 3. **ICL Resource Chain Curriculum** (`icl_resource_chain_curriculum_explanation.md`)
- **Focus**: In-context learning and sequential reasoning
- **Environment**: Dynamic resource conversion chains
- **Skills**: Adaptive learning, chain reasoning, resource optimization
- **Status**: Research/experimental curriculum
- **Usage**: `./tools/run.py experiments.recipes.icl_resource_chain.train`

### 4. **Arena Basic Easy Shaped** (`arena_basic_easy_shaped_curriculum_explanation.md`)
- **Focus**: Simplified multi-agent training
- **Environment**: Easier version of arena with shaped rewards
- **Skills**: Core arena mechanics with gentler learning curve
- **Purpose**: Beginner training and ablation studies
- **Usage**: `./tools/run.py experiments.recipes.arena_basic_easy_shaped.train`

## Evaluation Suites

### 5. **Systematic Exploration Memory** (`systematic_exploration_memory_curriculum_explanation.md`)
- **Focus**: Exploration strategies and spatial memory
- **Environment**: Specialized navigation evaluation maps
- **Purpose**: Benchmarking exploration and memory capabilities
- **Type**: Evaluation suite (not training curriculum)
- **Usage**: `./tools/run.py experiments.evals.systematic_exploration_memory.evaluate`

## Curriculum Comparison

| Curriculum             | Primary Focus        | Agent Count | Environment Type    | Difficulty Level |
| ---------------------- | -------------------- | ----------- | ------------------- | ---------------- |
| **Arena**              | Social Intelligence  | 24          | Resource Management | High             |
| **Navigation**         | Spatial Intelligence | 4           | Mazes & Terrain     | Medium           |
| **ICL Resource Chain** | Sequential Reasoning | Variable    | Dynamic Chains      | High             |
| **Arena Basic Easy**   | Core Mechanics       | 24          | Simplified Arena    | Low-Medium       |
| **Sys Exp Memory**     | Exploration Eval     | 1           | Specialized Maps    | Variable         |

## Research Applications

### Core Research Tracks:
1. **Multi-Agent Cooperation** → Arena curricula
2. **Spatial Intelligence** → Navigation curriculum
3. **In-Context Learning** → ICL Resource Chain
4. **Learning Efficiency** → Arena Basic Easy Shaped
5. **Exploration Strategies** → Systematic Exploration Memory

### Training Progression:
- **Start**: Arena Basic Easy Shaped (easier learning)
- **Foundation**: Navigation (spatial skills)
- **Core**: Arena (social intelligence)
- **Advanced**: ICL Resource Chain (meta-learning)

## File Organization

```
experiments/recipes/
├── arena.py                           # Main arena curriculum
├── navigation.py                      # Navigation curriculum
├── icl_resource_chain.py             # ICL curriculum
├── arena_basic_easy_shaped.py        # Simplified arena
└── scratchpad/example.py             # Development examples

experiments/evals/
├── navigation.py                     # Navigation evaluation
├── icl_resource_chain.py             # ICL evaluation
├── systematic_exploration_memory.py  # Exploration evaluation
└── registry.py                       # Evaluation registry
```

## Getting Started

### Primary Research Path:
1. Start with **Arena Basic Easy Shaped** for initial experiments
2. Move to **Navigation** for spatial intelligence foundation
3. Advance to main **Arena** curriculum for full multi-agent research
4. Explore **ICL Resource Chain** for advanced meta-learning research

### Quick Start Commands:
```bash
# Beginner-friendly training
./tools/run.py experiments.recipes.arena_basic_easy_shaped.train --args run=my_first_experiment

# Spatial intelligence training
./tools/run.py experiments.recipes.navigation.train --args run=my_navigation_experiment

# Full multi-agent research
./tools/run.py experiments.recipes.arena.train --args run=my_main_experiment

# Exploration evaluation
./tools/run.py experiments.evals.systematic_exploration_memory.evaluate --args policy_uri=wandb://run/my_experiment
```

Each curriculum includes detailed evaluation suites and supports both local development and distributed training through the Metta AI framework.
