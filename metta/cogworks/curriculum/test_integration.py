#!/usr/bin/env python3
"""Integration test for learning progress curriculum."""

import tempfile

import yaml
from adapter import create_learning_progress_curriculum, load_learning_progress_from_yaml


def test_adapter():
    """Test the adapter functionality."""
    print("Testing learning progress curriculum adapter...")

    # Test dict-based creation
    config_dict = {
        "ema_timescale": 0.01,
        "progress_smoothing": 0.05,
        "rand_task_rate": 0.25,
        "memory": 10,
    }

    curriculum = create_learning_progress_curriculum(config_dict)
    assert curriculum is not None
    print("✅ Dict-based creation works")

    # Test YAML-based creation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        yaml_path = f.name

    try:
        curriculum = load_learning_progress_from_yaml(yaml_path)
        assert curriculum is not None
        print("✅ YAML-based creation works")
    finally:
        import os

        os.unlink(yaml_path)

    # Test basic functionality
    task = curriculum.get_task()
    task.complete(0.5)
    lp = task.get_learning_progress()
    assert isinstance(lp, float)
    print(f"✅ Learning progress: {lp:.3f}")

    stats = curriculum.stats()
    assert isinstance(stats, dict)
    print(f"✅ Statistics: {stats}")

    print("🎉 Integration test passed!")


if __name__ == "__main__":
    test_adapter()
