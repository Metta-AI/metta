"""Training utilities for meta-analysis models."""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .models import MetaAnalysisModel

# Optional imports for wandb and visualization
try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
    plt = None

logger = logging.getLogger(__name__)


class TrainingCurveDataset(Dataset):
    """Dataset for training curve prediction."""

    def __init__(self, data_path: str, env_features: List[str], agent_features: List[str]):
        self.df = pd.read_csv(data_path)
        self.env_features = env_features
        self.agent_features = agent_features

        # Normalize features
        self.env_scaler = self._fit_scaler(self.df[env_features])
        self.agent_scaler = self._fit_scaler(self.df[agent_features])

    def _fit_scaler(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Fit normalization scaler."""
        scaler = {}
        for col in data.columns:
            values = data[col].dropna()
            if len(values) > 0:
                scaler[col] = (values.min(), values.max() - values.min())
            else:
                scaler[col] = (0.0, 1.0)
        return scaler

    def _normalize(self, data: pd.DataFrame, scaler: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Normalize data using fitted scaler."""
        normalized = []
        for col in data.columns:
            min_val, range_val = scaler[col]
            if range_val == 0:
                normalized.append(np.zeros(len(data)))
            else:
                normalized.append((data[col].fillna(min_val) - min_val) / range_val)
        return np.column_stack(normalized)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a training sample."""

        row = self.df.iloc[idx]

        # Extract and normalize features
        env_data = self._normalize(row[self.env_features].to_frame().T, self.env_scaler)
        agent_data = self._normalize(row[self.agent_features].to_frame().T, self.agent_scaler)

        # Parse training curve
        curve = json.loads(row["training_curve"])
        curve = np.array(curve, dtype=np.float32)

        return (torch.FloatTensor(env_data[0]), torch.FloatTensor(agent_data[0]), torch.FloatTensor(curve))


class MetaAnalysisTrainer:
    """Trainer for meta-analysis models."""

    def __init__(
        self,
        model: MetaAnalysisModel,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        beta: float = 1.0,  # VAE KL loss weight
        curve_weight: float = 1.0,  # Reward prediction loss weight
        wandb_run=None,
        tsne_interval: int = 10,
        tsne_sample_size: int = 128,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Loss weights
        self.beta = beta
        self.curve_weight = curve_weight

        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.curve_loss = nn.MSELoss()

        # Wandb and visualization settings
        self.wandb_run = wandb_run
        self.tsne_interval = tsne_interval
        self.tsne_sample_size = tsne_sample_size

    def vae_loss(
        self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss components."""

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(recon_x, x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total VAE loss
        vae_loss = recon_loss + self.beta * kl_loss

        return vae_loss, recon_loss, kl_loss

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""

        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_curve_loss = 0.0

        for _batch_idx, (env_config, agent_config, target_curve) in enumerate(dataloader):
            # Move to device
            env_config = env_config.to(self.device)
            agent_config = agent_config.to(self.device)
            target_curve = target_curve.to(self.device)

            # Forward pass
            (env_recon, env_mu, env_logvar, agent_recon, agent_mu, agent_logvar, predicted_curve) = self.model(
                env_config, agent_config
            )

            # Compute losses
            env_vae_loss, env_recon_loss, env_kl_loss = self.vae_loss(env_recon, env_config, env_mu, env_logvar)
            agent_vae_loss, agent_recon_loss, agent_kl_loss = self.vae_loss(
                agent_recon, agent_config, agent_mu, agent_logvar
            )

            # Curve prediction loss
            curve_loss = self.curve_loss(predicted_curve, target_curve)

            # Total loss
            loss = env_vae_loss + agent_vae_loss + self.curve_weight * curve_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += (env_recon_loss + agent_recon_loss).item()
            total_kl_loss += (env_kl_loss + agent_kl_loss).item()
            total_curve_loss += curve_loss.item()

        # Average losses
        num_batches = len(dataloader)
        return {
            "total_loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "curve_loss": total_curve_loss / num_batches,
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""

        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_curve_loss = 0.0

        with torch.no_grad():
            for env_config, agent_config, target_curve in dataloader:
                # Move to device
                env_config = env_config.to(self.device)
                agent_config = agent_config.to(self.device)
                target_curve = target_curve.to(self.device)

                # Forward pass
                (env_recon, env_mu, env_logvar, agent_recon, agent_mu, agent_logvar, predicted_curve) = self.model(
                    env_config, agent_config
                )

                # Compute losses
                env_vae_loss, env_recon_loss, env_kl_loss = self.vae_loss(env_recon, env_config, env_mu, env_logvar)
                agent_vae_loss, agent_recon_loss, agent_kl_loss = self.vae_loss(
                    agent_recon, agent_config, agent_mu, agent_logvar
                )
                curve_loss = self.curve_loss(predicted_curve, target_curve)

                # Total loss
                loss = env_vae_loss + agent_vae_loss + self.curve_weight * curve_loss

                total_loss += loss.item()
                total_recon_loss += (env_recon_loss + agent_recon_loss).item()
                total_kl_loss += (env_kl_loss + agent_kl_loss).item()
                total_curve_loss += curve_loss.item()

        # Average losses
        num_batches = len(dataloader)
        return {
            "val_loss": total_loss / num_batches,
            "val_recon_loss": total_recon_loss / num_batches,
            "val_kl_loss": total_kl_loss / num_batches,
            "val_curve_loss": total_curve_loss / num_batches,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_path: Optional[str] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict[str, List[float]]:
        """Train the model."""

        train_losses = []
        val_losses = []

        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            train_losses.append(train_metrics)

            # Validate
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                val_losses.append(val_metrics)

                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}")

            # Wandb logging
            if self.wandb_run is not None:
                log_dict = {
                    "train/total_loss": train_metrics["total_loss"],
                    "train/recon_loss": train_metrics["recon_loss"],
                    "train/kl_loss": train_metrics["kl_loss"],
                    "train/curve_loss": train_metrics["curve_loss"],
                }
                if val_dataloader is not None:
                    log_dict.update(
                        {
                            "val/total_loss": val_metrics["val_loss"],
                            "val/recon_loss": val_metrics["val_recon_loss"],
                            "val/kl_loss": val_metrics["val_kl_loss"],
                            "val/curve_loss": val_metrics["val_curve_loss"],
                        }
                    )
                self.wandb_run.log(log_dict, step=epoch)

            # TSNE visualization
            if self.wandb_run is not None and epoch % self.tsne_interval == 0:
                if train_dataset is not None:
                    self._log_tsne(train_dataset, epoch, prefix="train")
                if val_dataset is not None:
                    self._log_tsne(val_dataset, epoch, prefix="val")

            # Save model
            if save_path and (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"{save_path}_epoch_{epoch + 1}.pt")

        # Save final model
        if save_path:
            torch.save(self.model.state_dict(), f"{save_path}_final.pt")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    def _log_tsne(self, dataset: Dataset, epoch: int, prefix: str = "train"):
        """Create and log TSNE visualization of latent spaces."""
        if TSNE is None or plt is None or self.wandb_run is None:
            return

        # Sample a subset of the dataset
        sample_size = min(self.tsne_sample_size, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)

        env_latents = []
        agent_latents = []

        self.model.eval()
        with torch.no_grad():
            for idx in indices:
                env_config, agent_config, _ = dataset[idx]
                env_config = env_config.unsqueeze(0).to(self.device)
                agent_config = agent_config.unsqueeze(0).to(self.device)

                # Get latent representations
                env_mu, agent_mu = self.model.encode_only(env_config, agent_config)
                env_latents.append(env_mu.cpu().numpy())
                agent_latents.append(agent_mu.cpu().numpy())

        env_latents = np.vstack(env_latents)
        agent_latents = np.vstack(agent_latents)

        # Create TSNE embeddings
        env_tsne = TSNE(n_components=2, random_state=42).fit_transform(env_latents)
        agent_tsne = TSNE(n_components=2, random_state=42).fit_transform(agent_latents)

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(env_tsne[:, 0], env_tsne[:, 1], alpha=0.7)
        ax1.set_title(f"Environment Latent Space (Epoch {epoch})")
        ax1.set_xlabel("TSNE 1")
        ax1.set_ylabel("TSNE 2")

        ax2.scatter(agent_tsne[:, 0], agent_tsne[:, 1], alpha=0.7)
        ax2.set_title(f"Agent Latent Space (Epoch {epoch})")
        ax2.set_xlabel("TSNE 1")
        ax2.set_ylabel("TSNE 2")

        plt.tight_layout()

        # Log to wandb
        self.wandb_run.log({f"{prefix}/tsne_latents": wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    def predict_curve(self, env_config: torch.Tensor, agent_config: torch.Tensor) -> torch.Tensor:
        """Predict reward curve for given configs."""

        self.model.eval()
        with torch.no_grad():
            env_config = env_config.to(self.device)
            agent_config = agent_config.to(self.device)

            # Get latent representations
            env_mu, agent_mu = self.model.encode_only(env_config, agent_config)

            # Predict curve
            predicted_curve = self.model.predict_curve(env_mu, agent_mu)

            return predicted_curve.cpu()

    def sample_latent_space(self, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the learned latent spaces."""

        # Sample from standard normal
        env_latent = torch.randn(num_samples, self.model.env_vae.fc_mu.out_features)
        agent_latent = torch.randn(num_samples, self.model.agent_vae.fc_mu.out_features)

        return env_latent, agent_latent
