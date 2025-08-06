"""
Notebook generation utilities for mechanistic interpretation analysis.
"""

from pathlib import Path


class AnalysisNotebook:
    """
    Generates Jupyter notebooks for analysis workflows.
    """

    def __init__(self, output_dir: str = "notebooks"):
        """
        Initialize the notebook generator.

        Args:
            output_dir: Directory to save generated notebooks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_activation_recording_notebook(self, policy_uri: str, environment: str) -> Path:
        """
        Generate a notebook for recording activations.

        Args:
            policy_uri: Wandb URI of policy to analyze
            environment: Environment name

        Returns:
            Path to generated notebook
        """
        notebook_content = f'''# Activation Recording Analysis

This notebook demonstrates how to record LSTM activations from a trained policy.

## Setup

```python
import sys
sys.path.append("../../src")

from metta.mettagrid.analysis import (
    PolicyLoader, 
    ActivationRecorder, 
    SequenceExtractor,
    ProceduralSequenceGenerator
)
import torch
import wandb
```

## Load Policy

```python
# Initialize policy loader
policy_loader = PolicyLoader()

# Load policy from wandb
policy_uri = "{policy_uri}"
policy = policy_loader.load_policy_from_wandb(policy_uri)

print(f"Loaded policy from {{policy_uri}}")
print(f"Policy info: {{policy_loader.get_policy_info(policy_uri)}}")
```

## Generate Sequences

```python
# Initialize sequence generators
sequence_extractor = SequenceExtractor(max_sequence_length=50)
procedural_generator = ProceduralSequenceGenerator()

# Extract sequences from policy (placeholder - needs environment)
# sequences = sequence_extractor.extract_sequences(policy, environment, num_sequences=100)

# Generate procedural sequences for different strategies
strategies = ["goal_seeking", "exploration", "resource_collection", "combat", "avoidance"]
all_sequences = []

for strategy in strategies:
    sequences = procedural_generator.generate_sequences(
        strategy, num_sequences=20, sequence_length=30
    )
    all_sequences.extend(sequences)
    
print(f"Generated {{len(all_sequences)}} sequences")
```

## Record Activations

```python
# Initialize activation recorder
recorder = ActivationRecorder(storage_dir="activations")

# Record activations
environment_name = "{environment}"
activations_data = recorder.record_activations(
    policy=policy,
    sequences=all_sequences,
    policy_uri=policy_uri,
    environment=environment_name
)

print(f"Recorded activations for {{len(activations_data['sequences'])}} sequences")

# Save activations
filepath = recorder.save_activations(activations_data)
print(f"Saved activations to {{filepath}}")
```

## Analyze Recorded Data

```python
# Load saved activations
loaded_data = recorder.load_activations(filepath)

print("Activation data summary:")
print(f"- Policy: {{loaded_data['policy_uri']}}")
print(f"- Environment: {{loaded_data['environment']}}")
print(f"- Sequences: {{len(loaded_data['sequences'])}}")
print(f"- Recorded at: {{loaded_data['recorded_at']}}")

# Examine activation structure
for sequence_id, sequence_data in list(loaded_data['activations'].items())[:3]:
    print(f"\\nSequence {{sequence_id}}:")
    for layer_name, layer_data in sequence_data['activations'].items():
        if layer_data.get('hidden') is not None:
            print(f"  {{layer_name}} hidden: {{layer_data['hidden'].shape}}")
        if layer_data.get('cell') is not None:
            print(f"  {{layer_name}} cell: {{layer_data['cell'].shape}}")
```
'''

        filepath = self.output_dir / f"activation_recording_{policy_uri.replace('/', '_')}.ipynb"
        self._write_notebook(notebook_content, filepath)
        return filepath

    def generate_sae_training_notebook(self, activations_file: str) -> Path:
        """
        Generate a notebook for training sparse autoencoders.

        Args:
            activations_file: Path to saved activations file

        Returns:
            Path to generated notebook
        """
        notebook_content = f'''# Sparse Autoencoder Training

This notebook demonstrates how to train sparse autoencoders on recorded activations.

## Setup

```python
import sys
sys.path.append("../../src")

from metta.mettagrid.analysis import (
    ActivationRecorder,
    SAEConfig,
    SAETrainer
)
import torch
import wandb
import matplotlib.pyplot as plt
```

## Load Activation Data

```python
# Load recorded activations
recorder = ActivationRecorder()
activations_data = recorder.load_activations("{activations_file}")

print("Loaded activation data:")
print(f"- Policy: {{activations_data['policy_uri']}}")
print(f"- Environment: {{activations_data['environment']}}")
print(f"- Sequences: {{len(activations_data['sequences'])}}")

# Examine activation structure
for sequence_id, sequence_data in list(activations_data['activations'].items())[:1]:
    print(f"\\nSample sequence {{sequence_id}}:")
    for layer_name, layer_data in sequence_data['activations'].items():
        if layer_data.get('hidden') is not None:
            hidden_shape = layer_data['hidden'].shape
            print(f"  {{layer_name}} hidden: {{hidden_shape}}")
            # Calculate input size for SAE
            input_size = hidden_shape[-1] if len(hidden_shape) > 1 else hidden_shape[0]
            print(f"  SAE input size: {{input_size}}")
```

## Configure SAE

```python
# Determine input size from activation data
# This is a placeholder - you'll need to calculate this from your data
input_size = 512  # Replace with actual size from your activations
bottleneck_size = 64  # Number of bottleneck neurons

# Create SAE configuration
sae_config = SAEConfig(
    input_size=input_size,
    bottleneck_size=bottleneck_size,
    sparsity_target=0.1,  # 10% active neurons
    l1_lambda=1e-3,
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=100,
    device="cpu"  # or "cuda" if available
)

print("SAE Configuration:")
print(f"- Input size: {{sae_config.input_size}}")
print(f"- Bottleneck size: {{sae_config.bottleneck_size}}")
print(f"- Sparsity target: {{sae_config.sparsity_target}}")
print(f"- L1 lambda: {{sae_config.l1_lambda}}")
```

## Train SAE

```python
# Initialize trainer with wandb logging
trainer = SAETrainer(
    config=sae_config,
    wandb_project="mettagrid-sae-analysis"
)

# Train SAE on LSTM activations
sae_results = trainer.train(
    activations_data=activations_data,
    layer_name="lstm"  # or whatever layer name you recorded
)

print("Training completed!")
print(f"- Final loss: {{sae_results['final_loss']:.4f}}")
print(f"- Final sparsity: {{sae_results['final_sparsity']:.4f}}")
print(f"- Active neurons: {{sae_results['num_active_neurons']}}")
```

## Analyze Results

```python
# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curve
ax1.plot(sae_results['train_losses'])
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

# Sparsity curve
ax2.plot(sae_results['sparsity_metrics'])
ax2.set_title("Sparsity Over Time")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Sparsity")
ax2.grid(True)

plt.tight_layout()
plt.show()

# Analyze active neurons
print(f"\\nActive neurons: {{sae_results['active_neurons']}}")
print(f"Number of active neurons: {{len(sae_results['active_neurons'])}}")
print(f"Active neuron ratio: {{len(sae_results['active_neurons']) / sae_config.bottleneck_size:.2f}}")
```

## Save Model

```python
# Save trained SAE
model_filepath = trainer.save_model(sae_results, "sae_model.pt")
print(f"Saved model to {{model_filepath}}")

# Test loading
loaded_results = trainer.load_model(model_filepath)
print("Model loaded successfully!")
print(f"- Model type: {{type(loaded_results['model'])}}")
print(f"- Active neurons: {{loaded_results['active_neurons']}}")
```
'''

        filepath = self.output_dir / "sae_training.ipynb"
        self._write_notebook(notebook_content, filepath)
        return filepath

    def generate_concept_analysis_notebook(self, sae_model_file: str, activations_file: str) -> Path:
        """
        Generate a notebook for concept analysis.

        Args:
            sae_model_file: Path to trained SAE model
            activations_file: Path to activations file

        Returns:
            Path to generated notebook
        """
        notebook_content = f'''# Concept Analysis

This notebook demonstrates how to analyze concepts discovered by sparse autoencoders.

## Setup

```python
import sys
sys.path.append("../../src")

from metta.mettagrid.analysis import (
    ActivationRecorder,
    SAETrainer,
    ConceptAnalyzer,
    ConceptSteerer
)
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

## Load Data

```python
# Load activations
recorder = ActivationRecorder()
activations_data = recorder.load_activations("{activations_file}")

# Load trained SAE
trainer = SAETrainer(config=None)  # Config will be loaded from file
sae_results = trainer.load_model("{sae_model_file}")

print("Loaded data:")
print(f"- Activations: {{len(activations_data['sequences'])}} sequences")
print(f"- SAE active neurons: {{len(sae_results['active_neurons'])}}")
```

## Analyze Concepts

```python
# Initialize concept analyzer
analyzer = ConceptAnalyzer(sae_results)

# Analyze concepts
concepts = analyzer.analyze_concepts(
    activations_data=activations_data,
    sequences=activations_data['sequences']
)

print(f"Discovered {{len(concepts)}} concepts")

# Display concept information
for i, concept in enumerate(concepts[:10]):  # Show first 10
    print(f"\\nConcept {{i+1}}: {{concept.name}}")
    print(f"  Type: {{concept.concept_type.value}}")
    print(f"  Description: {{concept.description}}")
    print(f"  Confidence: {{concept.confidence:.3f}}")
    print(f"  Active neurons: {{concept.active_neurons}}")
    print(f"  Activation pattern shape: {{concept.activation_pattern.shape}}")
```

## Visualize Concepts

```python
# Create concept visualizations
analyzer.visualize_concepts(concepts)

# Additional analysis
print("\\nConcept type distribution:")
concept_types = [c.concept_type.value for c in concepts]
type_counts = {{}}
for concept_type in concept_types:
    type_counts[concept_type] = type_counts.get(concept_type, 0) + 1

for concept_type, count in type_counts.items():
    print(f"  {{concept_type}}: {{count}}")

print("\\nConfidence statistics:")
confidences = [c.confidence for c in concepts]
print(f"  Mean confidence: {{np.mean(confidences):.3f}}")
print(f"  Std confidence: {{np.std(confidences):.3f}}")
print(f"  Min confidence: {{np.min(confidences):.3f}}")
print(f"  Max confidence: {{np.max(confidences):.3f}}")
```

## Concept Steering Analysis

```python
# Initialize concept steerer
# Note: This requires the original policy, which would need to be loaded
# steerer = ConceptSteerer(policy, sae_results)

# Example steering analysis (placeholder)
print("Concept steering analysis would be performed here")
print("This would involve:")
print("1. Loading the original policy")
print("2. Applying concept activations")
print("3. Testing behavior changes")
print("4. Comparing steering effects")
```

## Cross-Policy Analysis

```python
# This section would compare concepts across different policies
print("Cross-policy concept analysis would be performed here")
print("This would involve:")
print("1. Loading multiple policies")
print("2. Training SAEs for each")
print("3. Comparing discovered concepts")
print("4. Analyzing concept evolution")
```
'''

        filepath = self.output_dir / "concept_analysis.ipynb"
        self._write_notebook(notebook_content, filepath)
        return filepath

    def _write_notebook(self, content: str, filepath: Path):
        """Write notebook content to file."""
        with open(filepath, "w") as f:
            f.write(content)
