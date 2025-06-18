#!/usr/bin/env python3
"""
Demonstration of MettaAgent revisioning improvements.

This script showcases:
1. Migration of old checkpoints
2. Version compatibility checking
3. Compression utilities
4. Different save/load methods
"""

import argparse
import logging
import os
import tempfile

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demo_migration():
    """Demonstrate checkpoint migration."""
    print("\n=== Checkpoint Migration Demo ===")

    from metta.agent.migrate_checkpoints import CheckpointMigrator

    # Create a mock old checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        old_checkpoint_path = os.path.join(tmpdir, "old_model.pt")

        # Simulate an old PolicyRecord checkpoint
        class MockPolicyRecord:
            def __init__(self):
                self._policy = torch.nn.Linear(10, 10)
                self.name = "old_model"
                self.uri = "file://old_model.pt"
                self.metadata = {"epoch": 100, "score": 0.85}

        torch.save(MockPolicyRecord(), old_checkpoint_path)

        # Migrate it
        migrator = CheckpointMigrator(backup=True)
        success = migrator.migrate_checkpoint(old_checkpoint_path)

        if success:
            print("✓ Successfully migrated checkpoint")
            print(f"  Original: {old_checkpoint_path}")
            print(f"  Backup: {old_checkpoint_path}.backup")
        else:
            print("✗ Migration failed")

        migrator.print_summary()


def demo_version_compatibility():
    """Demonstrate version compatibility checking."""
    print("\n=== Version Compatibility Demo ===")

    from metta.agent.version_compatibility import VersionInfo, compatibility_checker

    # Simulate different version scenarios
    scenarios = [
        {
            "name": "Fully Compatible",
            "checkpoint": VersionInfo(
                checkpoint_format=2, observation_space="v1", action_space="v1", layer_architecture="v1"
            ),
            "runtime": VersionInfo(
                checkpoint_format=2, observation_space="v1", action_space="v1", layer_architecture="v1"
            ),
        },
        {
            "name": "Action Space Mismatch",
            "checkpoint": VersionInfo(
                checkpoint_format=2, observation_space="v1", action_space="v1", layer_architecture="v1"
            ),
            "runtime": VersionInfo(
                checkpoint_format=2,
                observation_space="v1",
                action_space="v2",  # Different action space
                layer_architecture="v1",
            ),
        },
        {
            "name": "Incompatible Format",
            "checkpoint": VersionInfo(
                checkpoint_format=3,  # Newer format
                observation_space="v2",
                action_space="v2",
                layer_architecture="v2",
            ),
            "runtime": VersionInfo(
                checkpoint_format=2,  # Older runtime
                observation_space="v1",
                action_space="v1",
                layer_architecture="v1",
            ),
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        report = compatibility_checker.check_compatibility(scenario["checkpoint"], scenario["runtime"])
        print(report)


def demo_compression():
    """Demonstrate checkpoint compression."""
    print("\n=== Checkpoint Compression Demo ===")

    from metta.agent.checkpoint_compression import CheckpointCompressor

    # Create a test checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        test_checkpoint = os.path.join(tmpdir, "test_model.pt")

        # Create a realistic-sized checkpoint
        checkpoint_data = {
            "model_state_dict": {
                f"layer_{i}": torch.randn(128, 128)
                for i in range(50)  # ~50MB uncompressed
            },
            "metadata": {"epoch": 100},
            "checkpoint_format_version": 2,
        }

        torch.save(checkpoint_data, test_checkpoint)
        original_size = os.path.getsize(test_checkpoint) / 1024 / 1024  # MB
        print(f"Original checkpoint size: {original_size:.1f} MB")

        # Test different compression methods
        for method in ["gzip", "lz4", "zstd"]:
            compressor = CheckpointCompressor(method)
            compressed_path = compressor.compress_file(test_checkpoint)
            compressed_size = os.path.getsize(compressed_path) / 1024 / 1024
            reduction = (1 - compressed_size / original_size) * 100

            print(f"\n{method.upper()}:")
            print(f"  Compressed size: {compressed_size:.1f} MB")
            print(f"  Reduction: {reduction:.1f}%")

            # Test decompression
            decompressed_path = compressor.decompress_file(compressed_path)

            # Verify integrity
            original_data = torch.load(test_checkpoint)
            decompressed_data = torch.load(decompressed_path)

            keys_match = set(original_data.keys()) == set(decompressed_data.keys())
            print(f"  Integrity check: {'✓ PASSED' if keys_match else '✗ FAILED'}")


def demo_save_load_methods():
    """Demonstrate different save/load methods."""
    print("\n=== Save/Load Methods Demo ===")

    from metta.agent.metta_agent import MettaAgent

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock agent
        mock_model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 10))

        agent = MettaAgent(model=mock_model, model_type="mock", name="demo_agent", metadata={"epoch": 50, "score": 0.9})

        # Method 1: Standard save/load
        print("\n1. Standard Save/Load:")
        standard_path = os.path.join(tmpdir, "standard.pt")
        agent.save(standard_path)
        standard_size = os.path.getsize(standard_path) / 1024  # KB
        print(f"   File size: {standard_size:.1f} KB")

        # Method 2: Training save/load
        print("\n2. Training Save/Load:")
        training_path = os.path.join(tmpdir, "training.pt")
        agent.save_for_training(training_path)
        training_size = os.path.getsize(training_path) / 1024
        print(f"   File size: {training_size:.1f} KB")
        print(f"   Size increase: {(training_size / standard_size - 1) * 100:.1f}%")

        # Method 3: Compressed save
        print("\n3. Compressed Save:")
        compressed_path = os.path.join(tmpdir, "compressed.pt")
        agent.save(compressed_path, compress="zstd")
        compressed_size = os.path.getsize(compressed_path + ".zst") / 1024
        print(f"   File size: {compressed_size:.1f} KB")
        print(f"   Reduction: {(1 - compressed_size / standard_size) * 100:.1f}%")

        # Load timing comparison
        print("\n4. Load Speed Comparison:")
        import time

        # Standard load (would fail without proper model reconstruction)
        try:
            start = time.time()
            loaded = MettaAgent.load(standard_path)
            standard_time = time.time() - start
            print(f"   Standard load: {standard_time * 1000:.1f} ms")
        except Exception:
            print("   Standard load: Failed (expected for mock model)")

        # Training load
        start = time.time()
        loaded = MettaAgent.load_for_training(training_path)
        training_time = time.time() - start
        print(f"   Training load: {training_time * 1000:.1f} ms")

        # Compressed load
        start = time.time()
        loaded = MettaAgent.load(compressed_path + ".zst")
        compressed_time = time.time() - start
        print(f"   Compressed load: {compressed_time * 1000:.1f} ms")


def demo_distributed_compatibility():
    """Demonstrate distributed training compatibility."""
    print("\n=== Distributed Training Compatibility ===")

    from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent

    # Create a mock agent
    mock_model = torch.nn.Linear(10, 10)
    agent = MettaAgent(model=mock_model, model_type="mock", name="distributed_demo")

    # Simulate distributed wrapper
    print("Testing DistributedMettaAgent delegation:")

    # Mock distributed setup
    import unittest.mock as mock

    with mock.patch("torch.nn.SyncBatchNorm.convert_sync_batchnorm", return_value=agent):
        with mock.patch("torch.nn.parallel.DistributedDataParallel.__init__", return_value=None):
            dist_agent = DistributedMettaAgent(agent, device="cpu")
            dist_agent.module = agent  # Manually set since we're mocking

    # Test method delegation
    tests = [
        ("key_and_version", lambda: dist_agent.key_and_version()),
        ("policy_as_metta_agent", lambda: dist_agent.policy_as_metta_agent()),
        ("Attribute access (name)", lambda: dist_agent.name),
        ("Attribute access (metadata)", lambda: dist_agent.metadata),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            print(f"  ✓ {test_name}: {result}")
        except Exception as e:
            print(f"  ✗ {test_name}: {e}")


def main():
    """Run all demonstrations."""
    parser = argparse.ArgumentParser(description="Demonstrate MettaAgent improvements")
    parser.add_argument(
        "--demo",
        choices=["all", "migration", "compatibility", "compression", "save_load", "distributed"],
        default="all",
        help="Which demo to run",
    )

    args = parser.parse_args()

    demos = {
        "migration": demo_migration,
        "compatibility": demo_version_compatibility,
        "compression": demo_compression,
        "save_load": demo_save_load_methods,
        "distributed": demo_distributed_compatibility,
    }

    if args.demo == "all":
        for demo_func in demos.values():
            demo_func()
    else:
        demos[args.demo]()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
