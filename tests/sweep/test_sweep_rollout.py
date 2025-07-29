"""Unit tests for sweep rollout orchestration."""

import logging
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
        """Create a mock configuration."""
        return DictConfig(
            {
                "sweep_name": "test_sweep",
                "sweep_dir": "/tmp/test_sweep",
                "data_dir": "/tmp/test_sweep/runs",
                "sweep": {
                    "name": "test_sweep",
                    "metric": "reward",
                },
            }
        )

    @patch("sweep_rollout.evaluate_rollout")
    @patch("sweep_rollout.train_for_run")
    @patch("sweep_rollout.prepare_sweep_run")
    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.OmegaConf.load")
    def test_run_single_rollout_success(
        self, mock_load, mock_run_once, mock_prepare, mock_train, mock_evaluate, mock_config
    ):
        """Test successful rollout execution."""
        # Mock run_once to execute the function
        mock_run_once.side_effect = lambda func: func()

        # Prepare returns run info
        mock_train_cfg = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/test_sweep/runs/test_sweep.r.1",
                "dist_cfg_path": "/tmp/test_sweep/runs/test_sweep.r.1/dist_cfg.yaml",
            }
        )
        mock_prepare.return_value = (
            "test_sweep.r.1",
            mock_train_cfg,
            {"trainer": {"lr": 0.005}},
        )

        # Mock loading the full config
        mock_full_config = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/test_sweep/runs/test_sweep.r.1",
                "device": "cpu",
                "wandb": {"enabled": True},
            }
        )
        mock_load.return_value = mock_full_config

        # Evaluate returns results
        mock_evaluate.return_value = {"score": 0.95}

        # Call function
        result = run_single_rollout(mock_config)

        # Assertions
        assert result == 0
        assert mock_run_once.call_count == 2  # prepare and evaluate
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once_with(
            mock_full_config,
            {"trainer": {"lr": 0.005}},
            metric="reward",
            sweep_name="test_sweep",
            logger=logging.getLogger("sweep_rollout"),
        )

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.prepare_sweep_run")
    @patch("sweep_rollout.logger")
    def test_run_single_rollout_missing_config_keys(self, mock_logger, mock_prepare, mock_run_once, mock_config):
        """Test error handling for missing configuration keys."""
        # Remove required key
        del mock_config["sweep_name"]

        # Call function and expect exception
        with pytest.raises(Exception, match="Missing key sweep_name"):
            run_single_rollout(mock_config)

    @patch("sweep_rollout.train_for_run")
    @patch("sweep_rollout.prepare_sweep_run")
    @patch("sweep_rollout.run_once")
    def test_run_single_rollout_training_failure(self, mock_run_once, mock_prepare, mock_train, mock_config):
        """Test handling of training failures."""
        # Mock run_once to execute the function
        mock_run_once.side_effect = lambda func: func()

        # Prepare returns run info
        mock_train_cfg = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/test_sweep/runs/test_sweep.r.1",
                "dist_cfg_path": "/tmp/test_sweep/runs/test_sweep.r.1/dist_cfg.yaml",
            }
        )
        mock_prepare.return_value = (
            "test_sweep.r.1",
            mock_train_cfg,
            {"trainer": {"lr": 0.005}},
        )

        # Training fails
        mock_train.side_effect = RuntimeError("Training failed")

        # Call function and expect exception
        with pytest.raises(RuntimeError, match="Training failed"):
            run_single_rollout(mock_config)

    @patch("sweep_rollout.evaluate_rollout")
    @patch("sweep_rollout.train_for_run")
    @patch("sweep_rollout.prepare_sweep_run")
    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.OmegaConf.load")
    def test_run_single_rollout_evaluation_failure(
        self, mock_load, mock_run_once, mock_prepare, mock_train, mock_evaluate, mock_config
    ):
        """Test handling of evaluation failures."""
        # Mock run_once to execute the function
        mock_run_once.side_effect = lambda func: func()

        # Prepare returns run info
        mock_train_cfg = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/test_sweep/runs/test_sweep.r.1",
                "dist_cfg_path": "/tmp/test_sweep/runs/test_sweep.r.1/dist_cfg.yaml",
            }
        )
        mock_prepare.return_value = (
            "test_sweep.r.1",
            mock_train_cfg,
            {"trainer": {"lr": 0.005}},
        )

        # Mock loading the full config
        mock_full_config = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/test_sweep/runs/test_sweep.r.1",
                "device": "cpu",
            }
        )
        mock_load.return_value = mock_full_config

        # Evaluation fails
        mock_evaluate.side_effect = RuntimeError("Evaluation failed")

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

        # Create mock config
        mock_config = DictConfig(
            {
                "run": "test_run",
                "run_dir": "/tmp/test_run",
                "data_dir": "/tmp/test_run",
                "dist_cfg_path": "/tmp/test_run/dist_cfg.yaml",
            }
        )

        # Call function
        train_for_run(
            run_name="test_run",
            train_job_cfg=mock_config,
            original_args=["--gpus=4", "--nodes=2", "wandb=off"],
            logger=Mock(),
        )

        # Verify subprocess was called
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]

        # Check command structure
        assert "./devops/train.sh" in call_args
        assert "run=test_run" in call_args
        assert "dist_cfg_path=/tmp/test_run/dist_cfg.yaml" in call_args
        assert "--gpus=4" in call_args
        assert "--nodes=2" in call_args

    @patch("subprocess.run")
    def test_train_for_run_filters_duplicate_args(self, mock_subprocess_run):
        """Test that duplicate arguments are filtered out."""
        # Setup mock
        mock_subprocess_run.return_value = Mock(returncode=0)

        # Create mock config
        mock_config = DictConfig(
            {
                "run": "test_run",
                "run_dir": "/tmp/test_run",
                "data_dir": "/tmp/test_run",
                "dist_cfg_path": "/tmp/test_run/dist_cfg.yaml",
            }
        )

        # Call function with duplicate args
        train_for_run(
            run_name="test_run",
            train_job_cfg=mock_config,
            original_args=["run=old_run", "--gpus=4", "dist_cfg_path=/old/path"],
            logger=Mock(),
        )

        # Verify subprocess was called
        call_args = mock_subprocess_run.call_args[0][0]

        # Should have filtered out duplicate run and dist_cfg_path
        assert call_args.count("run=test_run") == 1
        assert call_args.count("dist_cfg_path=/tmp/test_run/dist_cfg.yaml") == 1
        assert "run=old_run" not in call_args
        assert "dist_cfg_path=/old/path" not in call_args

    @patch("subprocess.run")
    def test_train_for_run_handles_failure(self, mock_subprocess_run):
        """Test error handling when training fails."""
        # Setup mock to fail
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "train.sh")

        # Create mock config
        mock_config = DictConfig(
            {
                "run": "test_run",
                "run_dir": "/tmp/test_run",
                "data_dir": "/tmp/test_run",
                "dist_cfg_path": "/tmp/test_run/dist_cfg.yaml",
            }
        )

        # Call function and expect exception
        with pytest.raises(Exception, match="Training failed for test_run"):
            train_for_run(
                run_name="test_run",
                train_job_cfg=mock_config,
                original_args=[],
                logger=Mock(),
            )

    @patch("subprocess.run")
    def test_train_for_run_without_logger(self, mock_subprocess_run):
        """Test that function works without a logger."""
        # Setup mock
        mock_subprocess_run.return_value = Mock(returncode=0)

        # Create mock config
        mock_config = DictConfig(
            {
                "run": "test_run",
                "run_dir": "/tmp/test_run",
                "data_dir": "/tmp/test_run",
                "dist_cfg_path": "/tmp/test_run/dist_cfg.yaml",
            }
        )

        # Call function without logger
        train_for_run(
            run_name="test_run",
            train_job_cfg=mock_config,
            original_args=[],
            logger=None,
        )

        # Should complete without error
        mock_subprocess_run.assert_called_once()
