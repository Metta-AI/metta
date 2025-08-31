# Systematic Exploration Memory Evaluation Suite

The **Systematic Exploration Memory evaluation suite** is a specialized collection of navigation tasks designed to test agents' systematic exploration strategies and spatial memory capabilities. While not a full training curriculum, it provides comprehensive benchmarks for measuring exploration and memory performance.

## What is the Systematic Exploration Memory Suite?

This evaluation suite consists of carefully designed navigation environments that test different aspects of systematic exploration and memory. Each environment challenges agents to develop efficient exploration strategies and remember spatial information across extended time periods.

## Test Environments Overview

The suite includes multiple specialized environments, each targeting specific exploration and memory skills:

### Core Environment Types:

#### 1. **Structured Exploration Tasks**
- **boxout**: Contained exploration within boundaries
- **choose_wisely**: Decision-making in exploration trade-offs
- **corners**: Systematic coverage of boundary areas
- **hall_of_mirrors**: Navigation with visual confusion elements

#### 2. **Memory-Intensive Tasks**
- **journey_home**: Return-to-start navigation requiring memory
- **little_landmark_hard/easy**: Landmark-based navigation
- **maze_explore**: Complex maze exploration patterns
- **memory_matrix**: Grid-based memory challenges

#### 3. **Strategic Exploration**
- **optimal_coverage**: Efficient area coverage strategies
- **resource_gathering_exploration**: Exploration with resource optimization
- **systematic_coverage**: Methodical area exploration
- **temporal_exploration**: Time-dependent exploration patterns

#### 4. **Advanced Challenges**
- **exploration_efficiency**: Optimizing exploration speed vs thoroughness
- **memory_guided_exploration**: Using memory to guide future exploration
- **spatial_reasoning_exploration**: Complex spatial relationships
- **systematic_memory_test**: Comprehensive memory assessment

## Environment Characteristics

### Single-Agent Focus:
- **1 agent per environment** (unlike multi-agent navigation curriculum)
- **Navigation-only actions**: move, rotate, get_items
- **Resource collection**: Heart rewards from altars
- **Time limits**: Varying step limits based on task complexity

### Map Design Philosophy:
- **ASCII-based maps**: Hand-crafted for specific evaluation goals
- **Controlled complexity**: Each map tests specific exploration/memory skills
- **Progressive difficulty**: From simple to complex spatial reasoning
- **Reproducible testing**: Fixed layouts for consistent evaluation

## Evaluation Metrics

### Exploration Metrics:
- **Area coverage**: Percentage of map explored
- **Exploration efficiency**: Resources collected per exploration time
- **Path optimality**: Directness of exploration paths
- **Backtracking frequency**: Memory utilization in navigation

### Memory Metrics:
- **Spatial recall**: Ability to remember explored areas
- **Landmark recognition**: Identification and use of key locations
- **Return navigation**: Finding way back to important locations
- **Pattern recognition**: Identifying exploration patterns

### Performance Benchmarks:
- **Completion rate**: Successfully finding all goals
- **Time efficiency**: Steps taken vs optimal path
- **Resource efficiency**: Hearts collected vs exploration time
- **Consistency**: Performance across multiple evaluation runs

## Usage in Research

### 1. **Exploration Strategy Analysis**
```bash
# Evaluate exploration strategies
./tools/run.py experiments.evals.systematic_exploration_memory.evaluate --args policy_uri=wandb://run/my_policy
```

### 2. **Memory Capability Assessment**
- Compare different agent architectures
- Measure spatial memory retention
- Analyze exploration pattern development

### 3. **Curriculum Learning Validation**
- Test transfer learning from navigation curriculum
- Validate exploration skill development
- Measure generalization across exploration tasks

## Relation to Main Curricula

| Aspect               | Systematic Exploration Memory   | Navigation Curriculum           | Arena Curriculum        |
| -------------------- | ------------------------------- | ------------------------------- | ----------------------- |
| **Primary Focus**    | Exploration & memory evaluation | General navigation training     | Multi-agent cooperation |
| **Agent Count**      | 1 agent                         | 4 agents                        | 24 agents               |
| **Environment Type** | Specialized evaluation maps     | Varied terrain training         | Resource management     |
| **Learning Goal**    | Exploration strategy assessment | Spatial intelligence foundation | Social intelligence     |
| **Task Structure**   | Fixed evaluation scenarios      | Progressive curriculum          | Dynamic multi-agent     |

## Research Applications

### 1. **Exploration Algorithm Research**
- Testing different exploration strategies (random, systematic, memory-guided)
- Comparing exploration efficiency across architectures
- Studying exploration pattern emergence

### 2. **Memory System Evaluation**
- Assessing spatial memory capabilities
- Measuring memory retention over time
- Analyzing memory utilization in navigation

### 3. **Benchmarking and Comparison**
- Standardized evaluation across different agent implementations
- Performance comparison for different training approaches
- Generalization testing across exploration scenarios

## File Structure

- `metta/experiments/evals/systematic_exploration_memory.py` - Evaluation suite configuration
- `metta/mettagrid/configs/maps/systematic_exploration_memory/` - ASCII map files
- Individual `.map` files for each evaluation environment

## Integration with Training

While primarily an evaluation suite, the Systematic Exploration Memory environments can inform training curriculum design by:

- **Identifying exploration weaknesses** in trained policies
- **Guiding curriculum development** for exploration skills
- **Validating transfer learning** from navigation training
- **Measuring exploration generalization** across scenarios

This evaluation suite serves as a critical tool for understanding and improving agents' exploration and memory capabilities within the broader Metta AI research framework.
