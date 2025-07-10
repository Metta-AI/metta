"""
Example test file demonstrating the usage of test markers for scheduled test runs.

This file shows how to properly categorize tests using pytest markers to control
when they run in the CI/CD pipeline.
"""

import time
from unittest.mock import Mock, patch

import pytest


@pytest.mark.hourly
def test_critical_agent_initialization():
    """
    Critical test that runs every hour to ensure agents can be initialized.

    This is a fast, critical path test that should catch major regressions quickly.
    """
    # Simulate a quick critical test
    assert True  # Replace with actual agent initialization test


@pytest.mark.hourly
def test_core_training_loop():
    """
    Verify the core training loop can execute without errors.

    Runs hourly to catch training pipeline breakages early.
    """
    # Simulate core functionality test
    mock_trainer = Mock()
    mock_trainer.train.return_value = {"loss": 0.5}
    result = mock_trainer.train()
    assert result["loss"] < 1.0


@pytest.mark.daily
def test_comprehensive_training_scenario():
    """
    Comprehensive end-to-end training test.

    This test runs daily and validates a complete training scenario.
    """
    # Simulate a more comprehensive test
    time.sleep(0.1)  # Simulate some work

    # Would include actual training logic here
    training_steps = 100
    final_loss = 0.1

    assert training_steps > 0
    assert final_loss < 0.5


@pytest.mark.daily
@pytest.mark.slow
def test_long_running_evaluation():
    """
    Long-running evaluation test that validates model performance.

    Marked as both daily and slow - runs in daily comprehensive suite.
    """
    # Simulate expensive computation
    time.sleep(0.2)

    # Would include actual evaluation logic
    eval_score = 0.95
    assert eval_score > 0.9


@pytest.mark.integration
def test_wandb_integration():
    """
    Test integration with Weights & Biases service.

    This test requires external service connectivity and is marked as integration.
    """
    with patch("wandb.init") as mock_wandb:
        mock_wandb.return_value = Mock()

        # Test wandb integration
        run = mock_wandb()
        assert run is not None


@pytest.mark.integration
@pytest.mark.daily
def test_s3_checkpoint_storage():
    """
    Test checkpoint storage to S3.

    Marked as both integration (requires AWS) and daily.
    """
    with patch("boto3.client") as mock_boto:
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3

        # Simulate S3 operations
        mock_s3.put_object.return_value = {"ETag": "12345"}

        result = mock_s3.put_object(Bucket="test", Key="checkpoint.pt", Body=b"data")
        assert "ETag" in result


@pytest.mark.slow
def test_memory_intensive_operation():
    """
    Memory-intensive test that is slow but doesn't need frequent runs.

    Only marked as slow - will run in daily comprehensive suite but not hourly.
    """
    # Simulate memory-intensive operation
    large_array = [0] * 1000000
    assert len(large_array) == 1000000


def test_regular_unit_test():
    """
    Regular unit test with no special markers.

    This runs on every PR/push as part of the standard CI pipeline.
    """
    assert 1 + 1 == 2


@pytest.mark.parametrize("num_agents", [1, 2, 4])
@pytest.mark.hourly
def test_multi_agent_scenario(num_agents):
    """
    Parameterized test that runs hourly for different agent counts.

    Combines parametrize with hourly marker for critical multi-agent tests.
    """
    assert num_agents > 0
    assert num_agents <= 4


# Example of conditional marking based on environment
if not pytest.importorskip("torch.cuda").is_available():
    pytestmark = pytest.mark.skip("CUDA required for GPU tests")

    @pytest.mark.daily
    def test_gpu_training():
        """This test will be skipped if CUDA is not available."""
        pass
