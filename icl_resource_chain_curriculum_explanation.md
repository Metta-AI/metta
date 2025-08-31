# ICL Resource Chain Curriculum in Metta AI

The **ICL Resource Chain curriculum** (In-Context Learning Resource Chain) is an advanced curriculum focused on developing agents' ability to learn and execute complex resource conversion chains through in-context learning. This curriculum emphasizes sequential reasoning, resource management, and adaptive learning in dynamic environments.

## What is the ICL Resource Chain Curriculum?

The ICL Resource Chain curriculum creates environments where agents must learn to navigate and utilize complex chains of resource converters. Unlike the static resource chains in the arena curriculum, this curriculum generates dynamic, variable-length conversion chains that test agents' ability to adapt and learn from context.

## Key Components:

### 1. **Dynamic Resource Chain Environment**
- **Procedural chain generation**: Automatically creates conversion chains of varying lengths
- **Multiple resource types**: ore_red, ore_blue, ore_green, battery variants, laser, blueprint, armor
- **Converter diversity**: Mines, generators, labs, factories, temples, lasers, altars
- **Chain complexity**: Variable chain lengths (configurable)
- **Sink management**: Multiple output targets that agents must satisfy

### 2. **Curriculum Learning System**
The ICL curriculum uses a sophisticated **converter chain task generator** that creates environments with:

#### Chain Structure:
```python
CONVERTER_TYPES = {
    "mine_red": empty_converters.mine_red,
    "mine_blue": empty_converters.mine_blue,
    "mine_green": empty_converters.mine_green,
    "generator_red": empty_converters.generator_red,
    "generator_blue": empty_converters.generator_blue,
    "generator_green": empty_converters.generator_green,
    "altar": empty_converters.altar,
    "lab": empty_converters.lab,
    "lasery": empty_converters.lasery,
    "factory": empty_converters.factory,
    "temple": empty_converters.temple,
}
```

#### Task Generation Process:
- **Chain lengths**: Configurable lengths (e.g., [2, 3, 4, 5])
- **Sink count**: Variable number of output targets
- **Resource flow**: Input → Converter → Output chains
- **Complexity scaling**: Progressive increase in chain complexity

### 3. **Training Dynamics**
- **Sequential reasoning**: Agents learn to follow multi-step conversion processes
- **Resource optimization**: Efficient use of limited resources across chains
- **Adaptive learning**: In-context learning from dynamic chain configurations
- **Goal decomposition**: Breaking down complex chains into manageable steps
- **Context awareness**: Understanding converter relationships and dependencies

### 4. **Evaluation and Adaptation**
- **Chain completion metrics**: Measuring successful conversion chain execution
- **Efficiency metrics**: Resource usage and time-to-completion
- **Adaptability scores**: Performance across different chain configurations
- **Generalization testing**: Performance on unseen chain structures

## Purpose

The ICL Resource Chain curriculum serves as a **testbed for in-context learning** and **sequential reasoning** in multi-agent systems. By exposing agents to variable conversion chains, it:

- **Develops adaptive intelligence**: Learning from context without extensive retraining
- **Tests generalization**: Performance across novel chain configurations
- **Measures reasoning capabilities**: Sequential planning and resource management
- **Advances meta-learning**: Learning to learn from dynamic environments

## Key Differences from Other Curricula

| Aspect          | ICL Resource Chain    | Arena                | Navigation         |
| --------------- | --------------------- | -------------------- | ------------------ |
| **Focus**       | Sequential reasoning  | Multi-agent dynamics | Spatial navigation |
| **Environment** | Dynamic chains        | Static resource web  | Mazes & terrains   |
| **Learning**    | In-context adaptation | Social cooperation   | Path optimization  |
| **Complexity**  | Chain dependencies    | Agent interactions   | Spatial complexity |
| **Resources**   | Conversion chains     | Resource gathering   | Goal collection    |

## Related Files in the Project

- `metta/experiments/recipes/icl_resource_chain.py` - Main ICL curriculum implementation
- `metta/experiments/evals/icl_resource_chain.py` - ICL evaluation suite
- `metta/metta/cogworks/curriculum/task_generator.py` - Task generation framework
- `metta/mettagrid/config/envs.py` - Environment configuration utilities

## Usage

While the ICL curriculum appears to be more experimental/research-oriented compared to the main arena and navigation curricula, it would follow similar patterns:

```python
# Training on ICL curriculum (hypothetical)
./tools/run.py experiments.recipes.icl_resource_chain.train --args run=my_icl_experiment

# Evaluating ICL performance
./tools/run.py experiments.recipes.icl_resource_chain.evaluate --args policy_uri=wandb://run/my_icl_experiment
```

## Research Applications

The ICL Resource Chain curriculum is particularly valuable for:

1. **Meta-learning research**: Testing agents' ability to learn from limited examples
2. **Sequential reasoning**: Understanding multi-step problem solving
3. **Resource optimization**: Complex resource allocation in dynamic systems
4. **General AI capabilities**: Measuring adaptability and context-awareness
5. **Curriculum learning**: Progressive difficulty in chain complexity
