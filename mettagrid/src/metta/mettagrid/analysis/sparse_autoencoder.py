"""
Sparse autoencoder implementation for mechanistic interpretation.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


@dataclass
class SAEConfig:
    """Configuration for sparse autoencoder training."""

    input_size: int
    bottleneck_size: int
    sparsity_target: float = 0.1  # Target sparsity (10% active neurons)
    l1_lambda: float = 1e-3  # L1 regularization strength
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    device: str = "cpu"


class SparseAutoencoder(nn.Module):
    """
    Linear sparse autoencoder for activation analysis.

    This implements a linear autoencoder with L1 regularization
    to encourage sparsity in the bottleneck layer.
    """

    def __init__(self, config: SAEConfig):
        """
        Initialize the sparse autoencoder.

        Args:
            config: Configuration for the autoencoder
        """
        super().__init__()
        self.config = config

        # Linear encoder and decoder
        self.encoder = nn.Linear(config.input_size, config.bottleneck_size)
        self.decoder = nn.Linear(config.bottleneck_size, config.input_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input activations [batch_size, input_size]

        Returns:
            Tuple of (reconstructed, bottleneck_activations)
        """
        # Encode
        bottleneck = self.encoder(x)

        # Decode
        reconstructed = self.decoder(bottleneck)

        return reconstructed, bottleneck

    def get_sparsity(self, bottleneck_activations: torch.Tensor) -> float:
        """
        Calculate sparsity of bottleneck activations.

        Args:
            bottleneck_activations: Activations from bottleneck layer

        Returns:
            Sparsity ratio (fraction of zero activations)
        """
        # Count non-zero activations
        non_zero = torch.count_nonzero(bottleneck_activations)
        total = bottleneck_activations.numel()

        return 1.0 - (non_zero.float() / total)

    def get_active_neurons(self, bottleneck_activations: torch.Tensor) -> List[int]:
        """
        Get indices of active neurons in bottleneck.

        Args:
            bottleneck_activations: Activations from bottleneck layer

        Returns:
            List of active neuron indices
        """
        # Get mean activation per neuron across batch
        mean_activations = torch.mean(torch.abs(bottleneck_activations), dim=0)

        # Find neurons with non-zero mean activation
        active_indices = torch.where(mean_activations > 0.01)[0].tolist()

        return active_indices


class SAETrainer:
    """
    Trainer for sparse autoencoders with wandb integration.
    """

    def __init__(self, config: SAEConfig, wandb_project: Optional[str] = None):
        """
        Initialize the SAE trainer.

        Args:
            config: SAE configuration
            wandb_project: Optional wandb project name for logging
        """
        self.config = config
        self.device = torch.device(config.device)

        # Initialize wandb if project specified
        self.wandb_run = None
        if wandb_project:
            self.wandb_run = wandb.init(
                project=wandb_project, config=config.__dict__, name=f"sae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def train(self, activations_data: Dict[str, Any], layer_name: str = "lstm") -> Dict[str, Any]:
        """
        Train a sparse autoencoder on activation data.

        Args:
            activations_data: Data from ActivationRecorder
            layer_name: Name of layer to train on

        Returns:
            Training results and trained model
        """
        # Prepare data
        activations = self._prepare_activations(activations_data, layer_name)

        # Create model
        model = SparseAutoencoder(self.config).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Training loop
        train_losses = []
        sparsity_metrics = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_sparsity = 0.0
            num_batches = 0

            # Create batches
            for i in range(0, len(activations), self.config.batch_size):
                batch = activations[i : i + self.config.batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)

                # Forward pass
                reconstructed, bottleneck = model(batch_tensor)

                # Calculate losses
                reconstruction_loss = nn.MSELoss()(reconstructed, batch_tensor)
                sparsity_loss = torch.mean(torch.abs(bottleneck))

                total_loss = reconstruction_loss + self.config.l1_lambda * sparsity_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Metrics
                epoch_loss += total_loss.item()
                epoch_sparsity += model.get_sparsity(bottleneck).item()
                num_batches += 1

            # Average metrics
            avg_loss = epoch_loss / num_batches
            avg_sparsity = epoch_sparsity / num_batches

            train_losses.append(avg_loss)
            sparsity_metrics.append(avg_sparsity)

            # Log to wandb
            if self.wandb_run:
                self.wandb_run.log(
                    {
                        "epoch": epoch,
                        "loss": avg_loss,
                        "sparsity": avg_sparsity,
                        "reconstruction_loss": reconstruction_loss.item(),
                        "sparsity_loss": sparsity_loss.item(),
                    }
                )

            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Sparsity={avg_sparsity:.4f}")

        # Get final active neurons
        with torch.no_grad():
            all_activations = torch.tensor(activations, dtype=torch.float32).to(self.device)
            _, bottleneck = model(all_activations)
            active_neurons = model.get_active_neurons(bottleneck)

        # Prepare results
        results = {
            "model": model,
            "config": self.config,
            "train_losses": train_losses,
            "sparsity_metrics": sparsity_metrics,
            "active_neurons": active_neurons,
            "final_sparsity": sparsity_metrics[-1],
            "final_loss": train_losses[-1],
            "num_active_neurons": len(active_neurons),
            "training_completed": datetime.now().isoformat(),
        }

        return results

    def _prepare_activations(self, activations_data: Dict[str, Any], layer_name: str) -> List[np.ndarray]:
        """
        Prepare activation data for training.

        Args:
            activations_data: Data from ActivationRecorder
            layer_name: Name of layer to extract activations from

        Returns:
            List of activation vectors
        """
        activations = []

        for _sequence_id, sequence_data in activations_data["activations"].items():
            layer_activations = sequence_data["activations"].get(layer_name, {})

            # Extract hidden state (preferred) or cell state
            if "hidden" in layer_activations and layer_activations["hidden"] is not None:
                hidden = layer_activations["hidden"]
                if isinstance(hidden, torch.Tensor):
                    activation_vector = hidden.flatten().cpu().numpy()
                else:
                    activation_vector = hidden.flatten()
            elif "cell" in layer_activations and layer_activations["cell"] is not None:
                cell = layer_activations["cell"]
                if isinstance(cell, torch.Tensor):
                    activation_vector = cell.flatten().cpu().numpy()
                else:
                    activation_vector = cell.flatten()
            else:
                # Skip if no valid activation found
                continue

            activations.append(activation_vector)

        if not activations:
            raise ValueError(f"No valid activations found for layer '{layer_name}'")

        # Ensure all activations have same size
        activation_size = len(activations[0])
        for _i, activation in enumerate(activations):
            if len(activation) != activation_size:
                raise ValueError(f"Inconsistent activation sizes: {len(activation)} vs {activation_size}")

        return activations

    def save_model(self, results: Dict[str, Any], filepath: Path) -> Path:
        """
        Save trained model and results.

        Args:
            results: Training results from train()
            filepath: Path to save model

        Returns:
            Path to saved model
        """
        # Save model state
        torch.save(
            {"model_state_dict": results["model"].state_dict(), "config": results["config"], "results": results},
            filepath,
        )

        # Save as wandb artifact if using wandb
        if self.wandb_run:
            artifact = wandb.Artifact(name=f"sae_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}", type="model")
            artifact.add_file(str(filepath))
            self.wandb_run.log_artifact(artifact)

        return filepath

    def load_model(self, filepath: Path) -> Dict[str, Any]:
        """
        Load trained model and results.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded results dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Reconstruct model
        model = SparseAutoencoder(checkpoint["config"]).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Update results with loaded model
        results = checkpoint["results"]
        results["model"] = model

        return results
