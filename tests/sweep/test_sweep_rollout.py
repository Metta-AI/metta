"""Unit tests for sweep rollout orchestration."""

import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig

# Import after sys.path manipulation to handle the tools directory
sys.path.insert(0, "tools")
from sweep_rollout import (
    main,
    run_single_rollout,
    train_for_run,
)


class TestMain:
    """Test the main entry point function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = {
            "sweep_name": "test_sweep",
            "sweep_server_uri": "http://test-server",
            "wandb": {
                "entity": "test_entity",
                "project": "test_project",
            },
            "sweep": {
                "metric": "reward",
                "goal": "maximize",
            },
            "runs_dir": "/tmp/test_sweep/runs",
            "sweep_dir": "/tmp/test_sweep",
            "rollout_retry_delay": 5,
            "max_consecutive_failures": 3,
        }
        return DictConfig(config)

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.run_single_rollout")
    @patch("sweep_rollout.setup_sweep")
    def test_main_single_node_execution(self, mock_setup_sweep, mock_run_single_rollout, mock_run_once, mock_config):
        """Test main function in single-node execution."""
        # Setup mocks
        mock_setup_sweep.return_value = "test_sweep_id_123"

        # run_once should execute the function and return its result
        mock_run_once.side_effect = lambda func, **kwargs: func()

        # Make run_single_rollout raise exception after 2 successful runs
        mock_run_single_rollout.side_effect = [None, None, KeyboardInterrupt()]

        # Patch sys.argv
        with patch.object(sys, "argv", ["sweep_rollout.py", "sweep_name=test_sweep"]):
            # Call main - it should catch KeyboardInterrupt and exit gracefully
            with pytest.raises(KeyboardInterrupt):
                with patch("sweep_rollout.hydra.main", lambda **kwargs: lambda func: func):
                    main(mock_config)

        # Verify setup was called
        assert mock_run_once.call_count >= 1
        assert mock_run_single_rollout.call_count == 3

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.run_single_rollout")
    @patch("sweep_rollout.time.sleep")
    def test_main_retry_logic_on_failures(self, mock_sleep, mock_run_single_rollout, mock_run_once, mock_config):
        """Test retry logic when rollouts fail."""
        # Setup mocks
        mock_run_once.side_effect = lambda func, **kwargs: func()

        # Simulate failures then success
        mock_run_single_rollout.side_effect = [
            Exception("Failed 1"),
            Exception("Failed 2"),
            Exception("Failed 3"),
            Exception("Max failures reached"),
        ]

        # Mock setup_sweep
        with patch("sweep_rollout.setup_sweep", return_value="test_sweep_id"):
            with patch.object(sys, "argv", ["sweep_rollout.py"]):
                with patch("sweep_rollout.hydra.main", lambda **kwargs: lambda func: func):
                    # Should exit with 0 after max failures
                    result = main(mock_config)
                    assert result == 0

        # Verify sleep was called between failures (4 times, once after each failure)
        assert mock_sleep.call_count == 4
        mock_sleep.assert_called_with(5)  # rollout_retry_delay

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.logger")
    def test_main_setup_failure(self, mock_logger, mock_run_once, mock_config):
        """Test handling of sweep setup failure."""

        # Make setup fail
        def setup_fail():
            raise Exception("Setup failed")

        mock_run_once.side_effect = lambda func, **kwargs: func()

        with patch("sweep_rollout.setup_sweep", side_effect=setup_fail):
            with patch.object(sys, "argv", ["sweep_rollout.py"]):
                with patch("sweep_rollout.hydra.main", lambda **kwargs: lambda func: func):
                    result = main(mock_config)
                    assert result == 1

        mock_logger.error.assert_called()


class TestRunSingleRollout:
    """Test the run_single_rollout function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return DictConfig(
            {
                "sweep": {"metric": "reward"},
                "trainer": {"batch_size": 2048},
                "wandb": {"entity": "test_entity", "project": "test_project"},
                "sweep_id": "test_sweep_123",
                "sweep_name": "test_sweep",
                "runs_dir": "/tmp/runs",
                "sweep_train_job": {"trainer": {"lr": 0.001}},
            }
        )

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.train_for_run")
    @patch("sweep_rollout.prepare_sweep_run")
    @patch("sweep_rollout.evaluate_rollout")
    def test_run_single_rollout_success(self, mock_evaluate, mock_prepare, mock_train, mock_run_once, mock_config):
        """Test successful single rollout execution."""
        # Setup mocks
        mock_run_once.side_effect = lambda func, **kwargs: func()

        # Prepare returns run info
        mock_prepare.return_value = (
            "test_sweep.r.1",
            DictConfig({"dist_cfg_path": "/tmp/dist.yaml", "data_dir": "/tmp/data"}),
            {"trainer": {"lr": 0.005}},
        )

        # Evaluate returns results
        mock_evaluate.return_value = {"score": 0.95}

        # Call function
        result = run_single_rollout(mock_config)

        # Assertions
        assert result == 0
        assert mock_run_once.call_count == 2  # prepare and evaluate
        mock_train.assert_called_once()

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.prepare_sweep_run")
    @patch("sweep_rollout.logger")
    def test_run_single_rollout_missing_config_keys(self, mock_logger, mock_prepare, mock_run_once, mock_config):
        """Test error handling for missing configuration keys."""
        # Remove required key
        del mock_config["sweep_id"]

        # Mock prepare_sweep_run to raise ValueError
        mock_prepare.side_effect = ValueError("Missing required configuration keys: ['sweep_id']")
        mock_run_once.side_effect = lambda func, **kwargs: func()

        # Call function and expect exception
        with pytest.raises(ValueError, match="Missing required configuration keys"):
            run_single_rollout(mock_config)

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.train_for_run")
    @patch("sweep_rollout.logger")
    def test_run_single_rollout_training_failure(self, mock_logger, mock_train, mock_run_once, mock_config):
        """Test handling of training failure."""
        # Setup mocks
        mock_run_once.side_effect = lambda func, **kwargs: func()

        # Prepare succeeds
        with patch("sweep_rollout.prepare_sweep_run") as mock_prepare:
            mock_prepare.return_value = (
                "test_sweep.r.1",
                DictConfig({"dist_cfg_path": "/tmp/dist.yaml", "data_dir": "/tmp/data"}),
                {},
            )

            # Training fails
            mock_train.side_effect = Exception("Training failed")

            # Call function and expect exception
            with pytest.raises(Exception, match="Training failed"):
                run_single_rollout(mock_config)

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.train_for_run")
    @patch("sweep_rollout.logger")
    def test_run_single_rollout_evaluation_failure(self, mock_logger, mock_train, mock_run_once, mock_config):
        """Test handling of evaluation failure."""

        # Setup mocks
        def run_once_side_effect(func, **kwargs):
            # First call (prepare) succeeds
            if not hasattr(run_once_side_effect, "call_count"):
                run_once_side_effect.call_count = 0
            run_once_side_effect.call_count += 1

            if run_once_side_effect.call_count == 1:
                return func()
            else:
                # Second call (evaluate) returns None
                return None

        mock_run_once.side_effect = run_once_side_effect

        with patch("sweep_rollout.prepare_sweep_run") as mock_prepare:
            mock_prepare.return_value = (
                "test_sweep.r.1",
                DictConfig({"dist_cfg_path": "/tmp/dist.yaml", "data_dir": "/tmp/data"}),
                {},
            )

            # Call function and expect exception
            with pytest.raises(RuntimeError, match="Evaluation failed"):
                run_single_rollout(mock_config)


class TestTrainForRun:
    """Test the train_for_run function."""

    @patch("subprocess.run")
    def test_train_for_run_command_construction(self, mock_subprocess_run):
        """Test that training command is constructed correctly."""
        # Setup mock
        mock_subprocess_run.return_value = Mock(returncode=0)

        # Call function
        train_for_run(
            run_name="test_run",
            dist_cfg_path="/tmp/dist.yaml",
            data_dir="/tmp/data",
            original_args=["--gpus=4", "--nodes=2", "wandb=off"],
            logger=Mock(),
        )

        # Verify command
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]

        assert cmd[0] == "./devops/train.sh"
        assert "run=test_run" in cmd
        assert "dist_cfg_path=/tmp/dist.yaml" in cmd
        assert "data_dir=/tmp/data" in cmd
        assert "--gpus=4" in cmd
        assert "--nodes=2" in cmd
        assert "wandb=off" in cmd

    @patch("subprocess.run")
    def test_train_for_run_filters_duplicate_args(self, mock_subprocess_run):
        """Test that duplicate arguments are filtered out."""
        # Setup mock
        mock_subprocess_run.return_value = Mock(returncode=0)

        # Call function with args that should be filtered
        train_for_run(
            run_name="test_run",
            dist_cfg_path="/tmp/dist.yaml",
            data_dir="/tmp/data",
            original_args=[
                "run=old_run",  # Should be filtered
                "sweep_name=old_sweep",  # Should be filtered
                "--gpus=4",  # Should be kept
            ],
        )

        # Verify command
        cmd = mock_subprocess_run.call_args[0][0]

        # Check filtered args are not present
        assert "run=old_run" not in cmd
        assert "sweep_name=old_sweep" not in cmd

        # Check new args are present
        assert "run=test_run" in cmd
        assert "--gpus=4" in cmd

    @patch("subprocess.run")
    def test_train_for_run_handles_failure(self, mock_subprocess_run):
        """Test handling of training subprocess failure."""
        # Setup mock to fail
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "train.sh")

        # Call function and expect exception
        with pytest.raises(Exception, match="Training failed for test_run"):
            train_for_run(
                run_name="test_run",
                dist_cfg_path="/tmp/dist.yaml",
                data_dir="/tmp/data",
                logger=Mock(),
            )

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_train_for_run_without_logger(self, mock_print, mock_subprocess_run):
        """Test that function works without logger (uses print)."""
        # Setup mock
        mock_subprocess_run.return_value = Mock(returncode=0)

        # Call function without logger
        train_for_run(
            run_name="test_run",
            dist_cfg_path="/tmp/dist.yaml",
            data_dir="/tmp/data",
        )

        # Verify print was called
        mock_print.assert_called()
        print_output = str(mock_print.call_args[0][0])
        assert "[SWEEP:test_run]" in print_output
        assert "Running:" in print_output
