#!/usr/bin/env python3
"""
Demo script showing the complete world model workflow:
1. Create synthetic observation data
2. Train world model on the data
3. Load and test the trained model
"""

import os

# Add current directory to path to import our modules
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))

from train_world_model import ObservationDataset, WorldModel, create_data_loaders


def create_synthetic_dataset(dataset_path: str, n_observations: int = 5000):
    """Create a synthetic dataset for testing."""
    print(f"Creating synthetic dataset with {n_observations} observations...")

    with h5py.File(dataset_path, "w") as f:
        dataset = f.create_dataset(
            "observations", shape=(n_observations, 200, 3), dtype=np.float32, chunks=True, compression="gzip"
        )

        # Create synthetic but structured observations
        # This simulates agent observations with some patterns
        for i in range(n_observations):
            # Create structured observation: agent position + surroundings
            obs = np.zeros((200, 3), dtype=np.float32)

            # Agent position (first few entries)
            obs[:5, 0] = np.random.uniform(-1, 1, 5)  # x positions
            obs[:5, 1] = np.random.uniform(-1, 1, 5)  # y positions
            obs[:5, 2] = np.random.choice([0, 1], 5)  # agent type

            # Environment features (remaining entries)
            # Add some spatial structure
            spatial_pattern = np.sin(np.linspace(0, 4 * np.pi, 195)) * np.random.uniform(0.1, 0.5)
            obs[5:, 0] = spatial_pattern + np.random.normal(0, 0.1, 195)
            obs[5:, 1] = np.roll(spatial_pattern, 50) + np.random.normal(0, 0.1, 195)
            obs[5:, 2] = np.random.uniform(0, 1, 195)

            dataset[i] = obs

        f.attrs["max_observations"] = n_observations
        f.attrs["collection_interval"] = 1

    print(f"✓ Created dataset: {dataset_path}")


def train_model_demo(dataset_path: str, output_path: str, epochs: int = 20):
    """Demo the training process."""
    print(f"\nTraining world model for {epochs} epochs...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(dataset_path, batch_size=128)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = WorldModel(latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for _batch_idx, observations in enumerate(train_loader):
            observations = observations.to(device)

            optimizer.zero_grad()
            reconstructed = model(observations)
            loss = torch.nn.functional.mse_loss(reconstructed, observations)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for observations in val_loader:
                observations = observations.to(device)
                reconstructed = model(observations)
                loss = torch.nn.functional.mse_loss(reconstructed, observations)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1:2d}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Save model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs - 1,
        "loss": val_loss,
        "latent_dim": 64,
    }
    torch.save(checkpoint, output_path)
    print(f"✓ Model saved: {output_path}")

    return val_loss


def test_trained_model(model_path: str, dataset_path: str):
    """Test the trained model."""
    print("\nTesting trained model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = WorldModel(latent_dim=checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load some test data
    dataset = ObservationDataset(dataset_path)
    test_obs = dataset[0:5]  # First 5 observations
    test_obs = test_obs.to(device)

    with torch.no_grad():
        # Test encoding
        latents = model.encode(test_obs)
        print(f"✓ Encoded {test_obs.shape} -> {latents.shape}")

        # Test reconstruction
        reconstructed = model(test_obs)
        reconstruction_error = torch.nn.functional.mse_loss(reconstructed, test_obs)
        print(f"✓ Reconstruction error: {reconstruction_error.item():.6f}")

        # Show compression ratio
        original_size = test_obs.numel()
        compressed_size = latents.numel()
        compression_ratio = original_size / compressed_size
        print(f"✓ Compression ratio: {compression_ratio:.1f}x ({original_size} -> {compressed_size})")


def main():
    print("🚀 World Model Workflow Demo\n")

    # Use temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "demo_observations.h5")
        model_path = os.path.join(temp_dir, "demo_world_model.pt")

        try:
            # Step 1: Create synthetic data
            create_synthetic_dataset(dataset_path, n_observations=2000)

            # Step 2: Train world model
            final_loss = train_model_demo(dataset_path, model_path, epochs=10)

            # Step 3: Test trained model
            test_trained_model(model_path, dataset_path)

            print("\n🎉 Demo completed successfully!")
            print(f"Final validation loss: {final_loss:.6f}")
            print("\nNext steps:")
            print("1. Configure trainer with observation_collection.enabled=true")
            print("2. Run training to collect real observations")
            print("3. Use train_world_model.py to train on real data")
            print("4. Configure trainer with world_model.checkpoint_path")
            print("5. Run training with pre-trained world model")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
