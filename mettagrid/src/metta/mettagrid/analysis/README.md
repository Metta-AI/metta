# Mechanistic Interpretation Analysis

This module provides tools for mechanistic interpretation of trained policies in mettagrid using sparse autoencoders. The goal is to extract interpretable concepts from policy activations and understand how different training procedures affect concept representation.

## Overview

The analysis pipeline consists of several stages:

1. **Policy Loading**: Load trained policies from wandb checkpoint URIs
2. **Activation Recording**: Record LSTM activations at sequence completion points
3. **Sequence Generation**: Generate diverse sequences for analysis
4. **Sparse Autoencoder Training**: Train linear SAEs to discover sparse representations
5. **Concept Analysis**: Analyze discovered concepts and their properties
6. **Concept Steering**: Test behavioral changes by manipulating concept activations

## Project Structure

```
analysis/
├── __init__.py                 # Main module exports
├── policy_loader.py            # Policy loading from wandb
├── activation_recorder.py      # LSTM activation recording
├── sequence_generator.py       # Sequence extraction and generation
├── sparse_autoencoder.py       # SAE implementation and training
├── concept_analysis.py         # Concept discovery and steering
├── visualization/              # Visualization utilities
│   ├── __init__.py
│   ├── wandb_plots.py         # Wandb plotting utilities
│   └── notebooks.py           # Notebook generation
└── utils/                     # Utility functions
    ├── __init__.py
    ├── storage.py             # Data storage management
    └── metadata.py            # Analysis metadata management
```

## Quick Start

### Stage 1: Activation Recording

```python
from metta.mettagrid.analysis import (
    PolicyLoader, 
    ActivationRecorder, 
    ProceduralSequenceGenerator
)

# Load policy
policy_loader = PolicyLoader()
policy = policy_loader.load_policy_from_wandb("entity/project/run_id")

# Generate sequences
generator = ProceduralSequenceGenerator()
sequences = generator.generate_sequences("goal_seeking", num_sequences=50)

# Record activations
recorder = ActivationRecorder()
activations_data = recorder.record_activations(
    policy=policy,
    sequences=sequences,
    policy_uri="entity/project/run_id",
    environment="mettagrid"
)

# Save activations
filepath = recorder.save_activations(activations_data)
```

### Stage 2: Sparse Autoencoder Training

```python
from metta.mettagrid.analysis import SAEConfig, SAETrainer

# Configure SAE
config = SAEConfig(
    input_size=512,  # LSTM hidden size
    bottleneck_size=64,
    sparsity_target=0.1,
    l1_lambda=1e-3,
    num_epochs=100
)

# Train SAE
trainer = SAETrainer(config, wandb_project="mettagrid-sae")
sae_results = trainer.train(activations_data, layer_name="lstm")

# Save model
model_path = trainer.save_model(sae_results, "sae_model.pt")
```

### Stage 3: Concept Analysis

```python
from metta.mettagrid.analysis import ConceptAnalyzer

# Analyze concepts
analyzer = ConceptAnalyzer(sae_results)
concepts = analyzer.analyze_concepts(activations_data, sequences)

# Visualize concepts
analyzer.visualize_concepts(concepts)

# Print concept information
for concept in concepts[:5]:
    print(f"Concept: {concept.name}")
    print(f"  Type: {concept.concept_type.value}")
    print(f"  Description: {concept.description}")
    print(f"  Confidence: {concept.confidence:.3f}")
```

## Key Components

### PolicyLoader

Loads trained policies from wandb checkpoint URIs:

```python
loader = PolicyLoader()
policy = loader.load_policy_from_wandb("entity/project/run_id")
policy_info = loader.get_policy_info("entity/project/run_id")
```

### ActivationRecorder

Records LSTM activations at sequence completion:

```python
recorder = ActivationRecorder()
activations_data = recorder.record_activations(
    policy, sequences, policy_uri, environment
)
```

### Sequence Generation

Two types of sequence generation:

1. **SequenceExtractor**: Extract sequences from policy rollouts
2. **ProceduralSequenceGenerator**: Generate hand-crafted sequences

```python
# Extract from policy
extractor = SequenceExtractor()
sequences = extractor.extract_sequences(policy, environment, num_sequences=100)

# Generate procedural sequences
generator = ProceduralSequenceGenerator()
strategies = ["goal_seeking", "exploration", "combat", "avoidance"]
for strategy in strategies:
    sequences.extend(generator.generate_sequences(strategy, num_sequences=25))
```

### Sparse Autoencoder

Linear SAE with L1 regularization for sparsity:

```python
config = SAEConfig(
    input_size=512,
    bottleneck_size=64,
    sparsity_target=0.1,
    l1_lambda=1e-3
)

trainer = SAETrainer(config, wandb_project="mettagrid-sae")
results = trainer.train(activations_data, layer_name="lstm")
```

### Concept Analysis

Analyze discovered concepts and their properties:

```python
analyzer = ConceptAnalyzer(sae_results)
concepts = analyzer.analyze_concepts(activations_data, sequences)

# Concept types:
# - BEHAVIORAL: Goal-seeking, exploration, combat, avoidance
# - SPATIAL: Wall proximity, open areas, landmarks
# - TEMPORAL: Episode phases, sequence completion
# - OBJECT_BASED: Resource proximity, enemy detection
```

### Concept Steering

Test behavioral changes by manipulating concept activations:

```python
steerer = ConceptSteerer(policy, sae_results)

# Add concept activation
result = steerer.steer_behavior(concept, steering_type="add", strength=1.0)

# Clamp concept activation
result = steerer.steer_behavior(concept, steering_type="clamp", strength=0.0)
```

## Ground Truth Concepts

The analysis framework is designed to discover several types of concepts:

### Behavioral Concepts
- **Goal-seeking**: Moving toward objectives
- **Exploration**: Discovering new areas
- **Resource collection**: Gathering items
- **Combat**: Engaging enemies
- **Avoidance**: Retreating from threats

### Spatial Concepts
- **Wall proximity**: Near walls/obstacles
- **Open areas**: In spacious regions
- **Landmarks**: Near distinctive features

### Temporal Concepts
- **Episode start**: Beginning of episode
- **Mid-episode**: Middle phase
- **Episode end**: Completion phase

### Object-based Concepts
- **Battery proximity**: Near energy sources
- **Altar interaction**: Near altars
- **Enemy detection**: Near threats

## Success Metrics

### Stage 1 Success Criteria
- Can load any trained policy from wandb
- Can record activations for 100+ sequences per policy
- Storage system handles multiple policies efficiently

### Stage 2 Success Criteria
- Can identify and extract behavioral concepts
- Can demonstrate concept steering
- Can compare concepts across policies
- Can track concept evolution

## Usage Examples

### Complete Workflow

```python
# 1. Load policy and record activations
policy_loader = PolicyLoader()
policy = policy_loader.load_policy_from_wandb("entity/project/run_id")

generator = ProceduralSequenceGenerator()
sequences = []
for strategy in ["goal_seeking", "exploration", "combat"]:
    sequences.extend(generator.generate_sequences(strategy, num_sequences=30))

recorder = ActivationRecorder()
activations_data = recorder.record_activations(
    policy, sequences, "entity/project/run_id", "mettagrid"
)

# 2. Train sparse autoencoder
config = SAEConfig(input_size=512, bottleneck_size=64, sparsity_target=0.1)
trainer = SAETrainer(config, wandb_project="mettagrid-sae")
sae_results = trainer.train(activations_data, layer_name="lstm")

# 3. Analyze concepts
analyzer = ConceptAnalyzer(sae_results)
concepts = analyzer.analyze_concepts(activations_data, sequences)

# 4. Test steering
steerer = ConceptSteerer(policy, sae_results)
for concept in concepts[:3]:
    result = steerer.steer_behavior(concept, "add", 1.0)
    print(f"Steering {concept.name}: {result['behavior_results']}")
```

### Cross-Policy Analysis

```python
# Compare concepts across different policies
policies = [
    "entity/project/policy1",
    "entity/project/policy2", 
    "entity/project/policy3"
]

all_concepts = []
for policy_uri in policies:
    # Load and analyze each policy
    policy = policy_loader.load_policy_from_wandb(policy_uri)
    # ... record activations, train SAE, analyze concepts
    all_concepts.append(concepts)

# Compare concept similarities
# (Implementation depends on specific comparison metrics)
```

## Integration with Existing Tools

### Wandb Integration
- Policy loading from wandb artifacts
- SAE training tracked in wandb
- Model storage as wandb artifacts
- Visualization plots logged to wandb

### Observatory Integration
- Future integration with observatory for visualization
- Real-time concept analysis
- Interactive steering experiments

## Dependencies

- `torch`: PyTorch for neural networks
- `wandb`: Weights & Biases for experiment tracking
- `numpy`: Numerical computations
- `matplotlib`: Plotting
- `seaborn`: Statistical visualizations

## Future Work

### Stage 3: Advanced Features
- Cross-coder for low-dimensional dynamics
- Integration with observatory
- More sophisticated concept extraction
- Cross-policy concept comparison

### Research Directions
- Concept evolution during training
- Training procedure impact on concepts
- Interpretable policy design
- Behavioral control via concept steering

## Contributing

This is a research tool for mechanistic interpretation. Key areas for contribution:

1. **Environment Integration**: Connect with actual mettagrid environments
2. **Concept Extraction**: Improve concept discovery algorithms
3. **Steering Mechanisms**: Develop more sophisticated activation manipulation
4. **Visualization**: Enhanced concept visualization tools
5. **Evaluation**: Metrics for concept quality and steering effectiveness

## References

- Sparse Autoencoders for Mechanistic Interpretability
- Concept Learning in Neural Networks
- Behavioral Analysis in Reinforcement Learning
- Policy Interpretability Methods 