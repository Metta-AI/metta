#!/usr/bin/env python3
"""
Test script to verify the world model workflow components work correctly.
"""

import os
import tempfile

import h5py
import numpy as np
import torch


def test_world_model():
    """Test that the world model works with different latent dimensions."""
    from agent.src.metta.agent.world_model import WorldModel

    print("Testing WorldModel...")

    # Test with different latent dimensions
    for latent_dim in [32, 64, 128]:
        model = WorldModel(latent_dim=latent_dim)

        # Test with batch of observations
        batch_size = 4
        obs = torch.randn(batch_size, 200, 3)

        # Test encoding
        latent = model.encode(obs)
        assert latent.shape == (batch_size, latent_dim), f"Expected {(batch_size, latent_dim)}, got {latent.shape}"

        # Test decoding
        decoded = model.decode(latent)
        assert decoded.shape == (batch_size, 600), f"Expected {(batch_size, 600)}, got {decoded.shape}"

        # Test full forward pass
        reconstructed = model(obs)
        assert reconstructed.shape == obs.shape, f"Expected {obs.shape}, got {reconstructed.shape}"

        print(f"  ✓ latent_dim={latent_dim} works correctly")

    print("WorldModel tests passed!")


def test_observation_dataset():
    """Test that the HDF5 dataset handling works."""
    print("Testing observation dataset...")

    # Create a test HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        test_h5_path = f.name

    try:
        # Create test dataset
        n_observations = 100
        with h5py.File(test_h5_path, "w") as f:
            dataset = f.create_dataset("observations", shape=(n_observations, 200, 3), dtype=np.float32)
            # Fill with random data
            for i in range(n_observations):
                dataset[i] = np.random.randn(200, 3).astype(np.float32)

            f.attrs["max_observations"] = n_observations
            f.attrs["collection_interval"] = 1

        # Test reading the dataset
        import sys

        sys.path.append("/Users/dffarr/Desktop/metta")

        from train_world_model import ObservationDataset

        dataset = ObservationDataset(test_h5_path)
        assert len(dataset) == n_observations, f"Expected {n_observations}, got {len(dataset)}"

        # Test getting an item
        obs = dataset[0]
        assert obs.shape == (200, 3), f"Expected (200, 3), got {obs.shape}"
        assert obs.dtype == torch.float32, f"Expected torch.float32, got {obs.dtype}"

        print(f"  ✓ Dataset contains {len(dataset)} observations")
        print("Observation dataset tests passed!")

    finally:
        # Clean up
        if os.path.exists(test_h5_path):
            os.unlink(test_h5_path)


def test_checkpoint_saving_loading():
    """Test that world model checkpoints can be saved and loaded."""
    print("Testing checkpoint save/load...")

    from agent.src.metta.agent.world_model import WorldModel

    # Create model and some test data
    original_latent_dim = 64
    model1 = WorldModel(latent_dim=original_latent_dim)
    obs = torch.randn(2, 200, 3)

    # Test original model
    model1(obs)

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        checkpoint = {
            "model_state_dict": model1.state_dict(),
            "latent_dim": original_latent_dim,
            "epoch": 0,
            "loss": 0.5,
        }
        torch.save(checkpoint, checkpoint_path)

        # Load into new model
        model2 = WorldModel(latent_dim=original_latent_dim)
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint["model_state_dict"])

        # Test that outputs match
        model1.eval()
        model2.eval()
        with torch.no_grad():
            output1 = model1(obs)
            output2 = model2(obs)

        assert torch.allclose(output1, output2, atol=1e-6), "Loaded model doesn't match original"

        print("  ✓ Checkpoint save/load works correctly")
        print("Checkpoint tests passed!")

    finally:
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)


def main():
    print("Running world model workflow tests...\n")

    try:
        test_world_model()
        print()
        test_observation_dataset()
        print()
        test_checkpoint_saving_loading()
        print()
        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
