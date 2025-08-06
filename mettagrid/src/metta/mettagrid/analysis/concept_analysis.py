"""
Concept analysis and steering utilities for mechanistic interpretation.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn


class ConceptType(Enum):
    """Types of concepts that can be analyzed."""

    BEHAVIORAL = "behavioral"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    OBJECT_BASED = "object_based"


@dataclass
class Concept:
    """Represents a discovered concept."""

    name: str
    concept_type: ConceptType
    active_neurons: List[int]
    activation_pattern: np.ndarray
    description: str
    confidence: float


class ConceptAnalyzer:
    """
    Analyzes concepts discovered by sparse autoencoders.

    This class provides tools for understanding what concepts
    are represented by different neurons in the sparse autoencoder.
    """

    def __init__(self, sae_results: Dict[str, Any]):
        """
        Initialize the concept analyzer.

        Args:
            sae_results: Results from SAE training
        """
        self.sae_results = sae_results
        self.model = sae_results["model"]
        self.active_neurons = sae_results["active_neurons"]

    def analyze_concepts(self, activations_data: Dict[str, Any], sequences: List[Dict[str, Any]]) -> List[Concept]:
        """
        Analyze concepts represented by active neurons.

        Args:
            activations_data: Recorded activation data
            sequences: Original sequences used for recording

        Returns:
            List of discovered concepts
        """
        concepts = []

        # Analyze each active neuron
        for neuron_idx in self.active_neurons:
            concept = self._analyze_neuron_concept(neuron_idx, activations_data, sequences)
            if concept:
                concepts.append(concept)

        return concepts

    def _analyze_neuron_concept(
        self, neuron_idx: int, activations_data: Dict[str, Any], sequences: List[Dict[str, Any]]
    ) -> Optional[Concept]:
        """
        Analyze what concept a specific neuron represents.

        Args:
            neuron_idx: Index of neuron to analyze
            activations_data: Recorded activation data
            sequences: Original sequences

        Returns:
            Discovered concept or None
        """
        # Get neuron activations across all sequences
        neuron_activations = []
        sequence_metadata = []

        for sequence_id, sequence_data in activations_data["activations"].items():
            # Get bottleneck activations for this sequence
            bottleneck_activations = self._get_bottleneck_activations(sequence_data, sequence_id)

            if bottleneck_activations is not None:
                neuron_activation = bottleneck_activations[neuron_idx]
                neuron_activations.append(neuron_activation)

                # Get sequence metadata
                seq_idx = int(sequence_id.split("_")[1])
                if seq_idx < len(sequences):
                    sequence_metadata.append(sequences[seq_idx])

        if not neuron_activations:
            return None

        # Analyze activation pattern
        activation_pattern = np.array(neuron_activations)

        # Determine concept type and characteristics
        concept_type, description, confidence = self._classify_concept(activation_pattern, sequence_metadata)

        return Concept(
            name=f"concept_{neuron_idx}",
            concept_type=concept_type,
            active_neurons=[neuron_idx],
            activation_pattern=activation_pattern,
            description=description,
            confidence=confidence,
        )

    def _get_bottleneck_activations(self, sequence_data: Dict[str, Any], sequence_id: str) -> Optional[np.ndarray]:
        """Get bottleneck activations for a sequence."""
        # This is a placeholder - actual implementation depends on
        # how bottleneck activations are stored
        # For now, we'll simulate bottleneck activations
        return np.random.randn(len(self.active_neurons))

    def _classify_concept(
        self, activation_pattern: np.ndarray, sequence_metadata: List[Dict[str, Any]]
    ) -> Tuple[ConceptType, str, float]:
        """
        Classify the type of concept represented by activation pattern.

        Args:
            activation_pattern: Activation values across sequences
            sequence_metadata: Metadata for each sequence

        Returns:
            Tuple of (concept_type, description, confidence)
        """
        # Simple heuristics for concept classification
        mean_activation = np.mean(activation_pattern)
        std_activation = np.std(activation_pattern)

        # Analyze sequence characteristics
        strategies = [seq.get("strategy", "unknown") for seq in sequence_metadata]
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Determine concept type based on activation characteristics
        if std_activation > 0.5:
            # High variance suggests behavioral concept
            return ConceptType.BEHAVIORAL, "Behavioral pattern", 0.8
        elif mean_activation > 0.3:
            # High mean suggests spatial concept
            return ConceptType.SPATIAL, "Spatial pattern", 0.7
        else:
            # Default to temporal concept
            return ConceptType.TEMPORAL, "Temporal pattern", 0.6

    def visualize_concepts(self, concepts: List[Concept], save_path: Optional[Path] = None):
        """
        Create visualizations of discovered concepts.

        Args:
            concepts: List of discovered concepts
            save_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Discovered Concepts Analysis")

        # Plot 1: Concept activation patterns
        ax1 = axes[0, 0]
        for concept in concepts[:5]:  # Show first 5 concepts
            ax1.plot(concept.activation_pattern, label=concept.name)
        ax1.set_title("Concept Activation Patterns")
        ax1.set_xlabel("Sequence Index")
        ax1.set_ylabel("Activation")
        ax1.legend()

        # Plot 2: Concept type distribution
        ax2 = axes[0, 1]
        concept_types = [c.concept_type.value for c in concepts]
        type_counts = {}
        for concept_type in concept_types:
            type_counts[concept_type] = type_counts.get(concept_type, 0) + 1
        ax2.bar(type_counts.keys(), type_counts.values())
        ax2.set_title("Concept Type Distribution")
        ax2.set_ylabel("Count")

        # Plot 3: Confidence distribution
        ax3 = axes[1, 0]
        confidences = [c.confidence for c in concepts]
        ax3.hist(confidences, bins=10)
        ax3.set_title("Concept Confidence Distribution")
        ax3.set_xlabel("Confidence")
        ax3.set_ylabel("Count")

        # Plot 4: Active neurons heatmap
        ax4 = axes[1, 1]
        activation_matrix = np.array([c.activation_pattern for c in concepts])
        sns.heatmap(activation_matrix, ax=ax4, cmap="viridis")
        ax4.set_title("Concept Activation Heatmap")
        ax4.set_xlabel("Sequence Index")
        ax4.set_ylabel("Concept Index")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()


class ConceptSteerer:
    """
    Steers policy behavior by manipulating concept activations.

    This class provides tools for adding or clamping activations
    to influence policy behavior based on discovered concepts.
    """

    def __init__(self, policy: nn.Module, sae_results: Dict[str, Any]):
        """
        Initialize the concept steerer.

        Args:
            policy: Policy to steer
            sae_results: Results from SAE training
        """
        self.policy = policy
        self.sae_results = sae_results
        self.model = sae_results["model"]
        self.active_neurons = sae_results["active_neurons"]

        # Store original policy state
        self.original_state = {}
        self._save_original_state()

    def _save_original_state(self):
        """Save original policy state for restoration."""
        for name, param in self.policy.named_parameters():
            self.original_state[name] = param.data.clone()

    def restore_original_state(self):
        """Restore policy to original state."""
        for name, param in self.policy.named_parameters():
            if name in self.original_state:
                param.data = self.original_state[name].clone()

    def add_concept_activation(self, concept_neurons: List[int], activation_strength: float = 1.0):
        """
        Add activation to specific concept neurons.

        Args:
            concept_neurons: Indices of neurons to activate
            activation_strength: Strength of activation to add
        """
        # This is a placeholder - actual implementation depends on
        # how to modify policy activations during forward pass
        # Would need to implement custom hooks or modify policy architecture
        print(f"Adding activation {activation_strength} to neurons {concept_neurons}")

    def clamp_concept_activation(self, concept_neurons: List[int], clamp_value: float = 0.0):
        """
        Clamp activation of specific concept neurons.

        Args:
            concept_neurons: Indices of neurons to clamp
            clamp_value: Value to clamp activations to
        """
        # This is a placeholder - actual implementation depends on
        # how to modify policy activations during forward pass
        print(f"Clamping neurons {concept_neurons} to {clamp_value}")

    def steer_behavior(self, concept: Concept, steering_type: str = "add", strength: float = 1.0) -> Dict[str, Any]:
        """
        Steer policy behavior using a specific concept.

        Args:
            concept: Concept to use for steering
            steering_type: "add" or "clamp"
            strength: Strength of steering effect

        Returns:
            Results of steering experiment
        """
        # Apply steering
        if steering_type == "add":
            self.add_concept_activation(concept.active_neurons, strength)
        elif steering_type == "clamp":
            self.clamp_concept_activation(concept.active_neurons, strength)
        else:
            raise ValueError(f"Unknown steering type: {steering_type}")

        # Test policy behavior (placeholder)
        behavior_results = self._test_policy_behavior()

        # Restore original state
        self.restore_original_state()

        return {
            "concept": concept.name,
            "steering_type": steering_type,
            "strength": strength,
            "behavior_results": behavior_results,
        }

    def _test_policy_behavior(self) -> Dict[str, Any]:
        """
        Test policy behavior after steering.

        Returns:
            Behavior metrics
        """
        # This is a placeholder - actual implementation would
        # run policy in environment and measure behavior
        return {
            "combativeness": np.random.random(),
            "exploration": np.random.random(),
            "goal_seeking": np.random.random(),
            "resource_collection": np.random.random(),
        }

    def compare_steering_effects(
        self,
        concepts: List[Concept],
        steering_types: List[str] = None,
        strengths: List[float] = None,
    ) -> Dict[str, Any]:
        if steering_types is None:
            steering_types = ["add", "clamp"]
        if strengths is None:
            strengths = [0.5, 1.0, 2.0]
        """
        Compare effects of different steering approaches.

        Args:
            concepts: Concepts to test
            steering_types: Types of steering to test
            strengths: Strengths to test

        Returns:
            Comparison results
        """
        results = {"concepts": [], "steering_types": [], "strengths": [], "behavior_metrics": []}

        for concept in concepts:
            for steering_type in steering_types:
                for strength in strengths:
                    steering_result = self.steer_behavior(concept, steering_type, strength)

                    results["concepts"].append(concept.name)
                    results["steering_types"].append(steering_type)
                    results["strengths"].append(strength)
                    results["behavior_metrics"].append(steering_result["behavior_results"])

        return results
