#!/usr/bin/env python3
"""
Standalone script to train the world model on collected observation data.

This script loads observation data from an HDF5 file and trains the world model
to reconstruct observations. It's designed to be independent and allow
experimentation with different architectures and hyperparameters.

Usage:
    python train_world_model.py --dataset observations_dataset.h5 --output world_model.pt
"""

import argparse
import logging
import os
import time
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Minimal world model implementation (standalone)
class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Encoder layers
        self.l1 = nn.Linear(200 * 3, 2048)
        self.l2 = nn.Linear(2048, 2048)
        self.l3 = nn.Linear(2048, 1024)
        self.l4 = nn.Linear(1024, latent_dim)

        # Decoder layers
        self.l5 = nn.Linear(latent_dim, 1024)
        self.l6 = nn.Linear(1024, 2048)
        self.l7 = nn.Linear(2048, 2048)
        self.l8 = nn.Linear(2048, 200 * 3)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        # Accept [B, 200, 3] or already-flattened [B, 600]
        if obs.dim() == 3:
            obs = obs.reshape(obs.shape[0], -1)
        # Ensure floating dtype for linear layers
        obs = obs.float()
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l5(latent))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = self.l8(x)
        return x

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Combined encode and decode for reconstruction loss training.
        """
        x = self.decode(self.encode(obs))
        # Return the same shape as input if input was [B, 200, 3]
        if obs.dim() == 3:
            x = x.view(obs.shape[0], 200, 3)
        return x


class ObservationDataset(Dataset):
    def __init__(self, h5_file_path: str):
        self.h5_file_path = h5_file_path
        # Open file to get size but don't keep it open
        with h5py.File(h5_file_path, "r") as f:
            self.length = f["observations"].shape[0]
            # Find actual data length (stop at first zero observation)
            observations = f["observations"]
            self.actual_length = self.length
            # Quick check to find actual data length
            for i in range(min(1000, self.length)):  # Check first 1000 entries
                if np.all(observations[i] == 0):
                    self.actual_length = i
                    break

            # If we didn't find zeros in the first 1000, check more sparsely
            if self.actual_length == self.length and self.length > 1000:
                step_size = max(1, self.length // 1000)
                for i in range(0, self.length, step_size):
                    if np.all(observations[i] == 0):
                        # Binary search to find exact boundary
                        start = max(0, i - step_size)
                        end = i
                        while start < end:
                            mid = (start + end) // 2
                            if np.all(observations[mid] == 0):
                                end = mid
                            else:
                                start = mid + 1
                        self.actual_length = start
                        break

        print(f"Dataset: {self.actual_length}/{self.length} observations contain data")

    def __len__(self):
        return self.actual_length

    def __getitem__(self, idx):
        # Open file each time (not efficient but simple and thread-safe)
        with h5py.File(self.h5_file_path, "r") as f:
            obs = f["observations"][idx]
        return torch.from_numpy(obs).float()


def create_data_loaders(dataset_path: str, batch_size: int, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    dataset = ObservationDataset(dataset_path)

    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def train_epoch(
    model: WorldModel, train_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, observations in enumerate(train_loader):
        observations = observations.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        reconstructed = model(observations)
        loss = F.mse_loss(reconstructed, observations)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")

    return total_loss / num_batches


def validate(model: WorldModel, val_loader: DataLoader, device: torch.device) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for observations in val_loader:
            observations = observations.to(device, non_blocking=True)

            # Forward pass
            reconstructed = model(observations)
            loss = F.mse_loss(reconstructed, observations)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def save_model(model: WorldModel, optimizer: torch.optim.Optimizer, epoch: int, loss: float, output_path: str) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "latent_dim": model.l4.out_features,
    }
    torch.save(checkpoint, output_path)
    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train world model on observation data")
    parser.add_argument("--dataset", required=True, help="Path to HDF5 dataset file")
    parser.add_argument("--output", default="world_model.pt", help="Output model path")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        return

    # Create data loaders
    logger.info("Loading dataset...")
    train_loader, val_loader = create_data_loaders(args.dataset, args.batch_size)
    logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Create model
    model = WorldModel(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_output_path = args.output.replace(".pt", "_best.pt")
            save_model(model, optimizer, epoch, val_loss, best_output_path)

        # Regular checkpoints
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = args.output.replace(".pt", f"_epoch_{epoch + 1}.pt")
            save_model(model, optimizer, epoch, val_loss, checkpoint_path)

    # Save final model
    save_model(model, optimizer, args.epochs - 1, val_loss, args.output)

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
