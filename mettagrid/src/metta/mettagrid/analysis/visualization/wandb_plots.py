"""
Wandb plotting utilities for mechanistic interpretation analysis.
"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import wandb


class WandbPlotter:
    """
    Creates and logs plots to wandb for analysis tracking.
    """

    def __init__(self, wandb_run=None):
        """
        Initialize the wandb plotter.

        Args:
            wandb_run: Optional wandb run to log to
        """
        self.wandb_run = wandb_run

    def log_sae_training_plots(self, sae_results: Dict[str, Any]):
        """
        Log sparse autoencoder training plots to wandb.

        Args:
            sae_results: Results from SAE training
        """
        if not self.wandb_run:
            return

        # Training loss plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sae_results["train_losses"])
        ax.set_title("SAE Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)

        self.wandb_run.log({"training_loss_plot": wandb.Image(fig)})
        plt.close(fig)

        # Sparsity plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sae_results["sparsity_metrics"])
        ax.set_title("SAE Sparsity Over Time")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Sparsity")
        ax.grid(True)

        self.wandb_run.log({"sparsity_plot": wandb.Image(fig)})
        plt.close(fig)

        # Active neurons histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        active_neurons = sae_results["active_neurons"]
        ax.hist(active_neurons, bins=20)
        ax.set_title("Active Neuron Distribution")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Count")

        self.wandb_run.log({"active_neurons_plot": wandb.Image(fig)})
        plt.close(fig)

    def log_concept_analysis_plots(self, concepts: List[Any]):
        """
        Log concept analysis plots to wandb.

        Args:
            concepts: List of discovered concepts
        """
        if not self.wandb_run:
            return

        # Concept activation patterns
        fig, ax = plt.subplots(figsize=(12, 8))
        for concept in concepts[:10]:  # Show first 10 concepts
            ax.plot(concept.activation_pattern, label=concept.name, alpha=0.7)
        ax.set_title("Concept Activation Patterns")
        ax.set_xlabel("Sequence Index")
        ax.set_ylabel("Activation")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True)

        self.wandb_run.log({"concept_activation_patterns": wandb.Image(fig)})
        plt.close(fig)

        # Concept type distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        concept_types = [c.concept_type.value for c in concepts]
        type_counts = {}
        for concept_type in concept_types:
            type_counts[concept_type] = type_counts.get(concept_type, 0) + 1
        ax.bar(type_counts.keys(), type_counts.values())
        ax.set_title("Concept Type Distribution")
        ax.set_ylabel("Count")

        self.wandb_run.log({"concept_type_distribution": wandb.Image(fig)})
        plt.close(fig)

        # Concept confidence distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        confidences = [c.confidence for c in concepts]
        ax.hist(confidences, bins=15)
        ax.set_title("Concept Confidence Distribution")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")

        self.wandb_run.log({"concept_confidence_distribution": wandb.Image(fig)})
        plt.close(fig)

    def log_steering_analysis_plots(self, steering_results: Dict[str, Any]):
        """
        Log steering analysis plots to wandb.

        Args:
            steering_results: Results from steering experiments
        """
        if not self.wandb_run:
            return

        # Behavior metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Steering Effects on Behavior")

        behavior_metrics = steering_results["behavior_metrics"]
        _concepts = steering_results["concepts"]
        _steering_types = steering_results["steering_types"]
        strengths = steering_results["strengths"]

        # Extract metrics
        combativeness = [m["combativeness"] for m in behavior_metrics]
        exploration = [m["exploration"] for m in behavior_metrics]
        goal_seeking = [m["goal_seeking"] for m in behavior_metrics]
        resource_collection = [m["resource_collection"] for m in behavior_metrics]

        # Plot each metric
        axes[0, 0].scatter(strengths, combativeness, alpha=0.6)
        axes[0, 0].set_title("Combativeness vs Steering Strength")
        axes[0, 0].set_xlabel("Steering Strength")
        axes[0, 0].set_ylabel("Combativeness")

        axes[0, 1].scatter(strengths, exploration, alpha=0.6)
        axes[0, 1].set_title("Exploration vs Steering Strength")
        axes[0, 1].set_xlabel("Steering Strength")
        axes[0, 1].set_ylabel("Exploration")

        axes[1, 0].scatter(strengths, goal_seeking, alpha=0.6)
        axes[1, 0].set_title("Goal Seeking vs Steering Strength")
        axes[1, 0].set_xlabel("Steering Strength")
        axes[1, 0].set_ylabel("Goal Seeking")

        axes[1, 1].scatter(strengths, resource_collection, alpha=0.6)
        axes[1, 1].set_title("Resource Collection vs Steering Strength")
        axes[1, 1].set_xlabel("Steering Strength")
        axes[1, 1].set_ylabel("Resource Collection")

        plt.tight_layout()

        self.wandb_run.log({"steering_effects": wandb.Image(fig)})
        plt.close(fig)

    def log_activation_comparison_plots(self, activations_data: Dict[str, Any], policy_names: List[str]):
        """
        Log activation comparison plots across policies.

        Args:
            activations_data: Activation data from multiple policies
            policy_names: Names of the policies
        """
        if not self.wandb_run:
            return

        # Activation pattern comparison
        fig, ax = plt.subplots(figsize=(12, 8))

        for _i, policy_name in enumerate(policy_names):
            # Extract activation patterns (placeholder)
            # This would need to be implemented based on actual data structure
            activation_pattern = np.random.randn(100)  # Placeholder
            ax.plot(activation_pattern, label=policy_name, alpha=0.7)

        ax.set_title("Activation Pattern Comparison Across Policies")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Activation")
        ax.legend()
        ax.grid(True)

        self.wandb_run.log({"activation_comparison": wandb.Image(fig)})
        plt.close(fig)
