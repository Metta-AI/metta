"""Test sweep rollout orchestration."""

import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig

# Add tools directory to path for imports
sys.path.insert(0, "tools")


class TestSweepRollout:
    """Test sweep rollout main functions."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return DictConfig(
            {
                "sweep_name": "test_sweep",
                "data_dir": "/tmp/data",
                "sim_name": "default_sim",
                "settings": {
                    "sweep_server_uri": "http://test-server",
                    "rollout_retry_delay": 5,
                    "max_consecutive_failures": 3,
                    "max_observations_to_load": 100,
                },
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                },
                "sweep": {
                    "metric": "reward",
                    "goal": "maximize",
                },
                "sweep_job_overrides": {
                    "trainer": {
                        "batch_size": 2048,
                    },
                },
            }
        )

    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.run_single_rollout")
    @patch("sweep_rollout.initialize_sweep")
    def test_main_successful_rollouts(self, mock_initialize, mock_run_single, mock_run_once, mock_config):
        """Test main function with successful rollouts."""
        from sweep_rollout import main

        # Setup mocks
        mock_run_once.side_effect = lambda func, **kwargs: func()
        mock_initialize.return_value = None

        # Simulate 3 successful rollouts then keyboard interrupt
        mock_run_single.side_effect = [0, 0, 0, KeyboardInterrupt()]

        # Run main
        with patch.object(sys, "argv", ["sweep_rollout.py"]):
            with pytest.raises(KeyboardInterrupt):
                main(mock_config)

        # Verify calls
        mock_initialize.assert_called_once()
        assert mock_run_single.call_count == 4

    @patch("sweep_rollout.time.sleep")
    @patch("sweep_rollout.run_once")
    @patch("sweep_rollout.run_single_rollout")
    @patch("sweep_rollout.initialize_sweep")
    def test_main_with_failures(self, mock_initialize, mock_run_single, mock_run_once, mock_sleep, mock_config):
        """Test main function handling failures and retries."""
        from sweep_rollout import main

        # Setup mocks
        mock_run_once.side_effect = lambda func, **kwargs: func()
        mock_initialize.return_value = None

        # Simulate failures then max consecutive failures
        mock_run_single.side_effect = [
            Exception("Failed"),
            Exception("Failed"),
            Exception("Failed"),
            Exception("Failed"),  # This triggers max consecutive failures
        ]

        # Run main
        with patch.object(sys, "argv", ["sweep_rollout.py"]):
            result = main(mock_config)

        # Should exit with code 1 after max failures
        assert result == 1
        assert mock_run_single.call_count == 4
        assert mock_sleep.call_count == 4

    @patch("sweep_rollout.OmegaConf.load")
    @patch("sweep_rollout.evaluate_sweep_rollout")
    @patch("sweep_rollout.launch_training_subprocess")
    @patch("sweep_rollout.prepare_sweep_run")
    @patch("sweep_rollout.run_once")
    def test_run_single_rollout_success(
        self,
        mock_run_once,
        mock_prepare,
        mock_launch,
        mock_evaluate,
        mock_load,
        mock_config,
    ):
        """Test successful single rollout execution."""
        from sweep_rollout import run_single_rollout

        # Setup mocks
        mock_run_once.side_effect = lambda func: func()
        mock_prepare.return_value = ("test_sweep.r.1", {"trainer": {"lr": 0.001}})
        mock_launch.return_value = Mock(returncode=0)

        # Mock config loading
        full_config = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/data/test_sweep.r.1",
                "wandb": {"entity": "test_entity", "project": "test_project"},
            }
        )
        mock_load.return_value = full_config

        # Mock evaluation
        mock_evaluate.return_value = {"reward": 0.85, "time.total": 100}

        # Run single rollout
        result = run_single_rollout(mock_config, original_args=["--gpus=4"])

        # Verify success
        assert result == 0
        mock_prepare.assert_called_once()
        mock_launch.assert_called_once()
        mock_evaluate.assert_called_once_with(
            full_config,
            {"trainer": {"lr": 0.001}},
            metric="reward",
            sweep_name="test_sweep",
        )

    @patch("sweep_rollout.prepare_sweep_run")
    @patch("sweep_rollout.run_once")
    def test_run_single_rollout_prepare_failure(self, mock_run_once, mock_prepare, mock_config):
        """Test handling of prepare_sweep_run failure."""
        from sweep_rollout import run_single_rollout

        # Setup mocks
        mock_run_once.side_effect = lambda func: func()
        mock_prepare.return_value = (None, None)

        # Run single rollout
        result = run_single_rollout(mock_config)

        # Should return error code
        assert result == 1

    @patch("subprocess.run")
    def test_launch_training_subprocess(self, mock_subprocess_run):
        """Test launch_training_subprocess command construction."""
        from sweep_rollout import launch_training_subprocess

        # Setup mock
        mock_run = Mock(returncode=0)
        subprocess.run.return_value = mock_run

        # Test config
        cfg = DictConfig(
            {
                "sweep_job_overrides": {
                    "trainer": {
                        "batch_size": 2048,
                    },
                },
                "sim_name": "navigation",
            }
        )

        # Call function
        launch_training_subprocess(
            run_name="test_run",
            protein_suggestion={"trainer": {"lr": 0.001}},
            sweep_name="test_sweep",
            wandb_entity="test_entity",
            wandb_project="test_project",
            cfg=cfg,
            original_args=["--gpus=4"],
        )

        # Verify subprocess was called
        subprocess.run.assert_called_once()
        call_args = subprocess.run.call_args[0][0]

        # Verify command structure
        assert "./devops/train.sh" in call_args
        assert "run=test_run" in call_args
        assert "wandb.entity=test_entity" in call_args
        assert "wandb.project=test_project" in call_args
        assert "wandb.group=test_sweep" in call_args
        assert "sim=navigation" in call_args
        assert any("trainer.lr" in arg for arg in call_args)
        assert any("trainer.batch_size" in arg for arg in call_args)
        assert "--gpus=4" in call_args

    @patch("subprocess.run")
    def test_launch_training_subprocess_failure(self, mock_subprocess_run):
        """Test launch_training_subprocess handling failure."""
        from sweep_rollout import launch_training_subprocess

        # Setup mock to fail
        subprocess.run.side_effect = subprocess.CalledProcessError(1, "train.sh")

        # Test config
        cfg = DictConfig(
            {
                "sweep_job_overrides": {},
                "sim_name": "default",
            }
        )

        # Call function and expect exception
        with pytest.raises(Exception, match="Training failed"):
            launch_training_subprocess(
                run_name="test_run",
                protein_suggestion={},
                sweep_name="test_sweep",
                wandb_entity="test_entity",
                wandb_project="test_project",
                cfg=cfg,
                original_args=[],
            )

    @patch("subprocess.run")
    def test_launch_training_subprocess_filters_duplicates(self, mock_subprocess_run):
        """Test that duplicate arguments are filtered."""
        from sweep_rollout import launch_training_subprocess

        # Setup mock
        mock_run = Mock(returncode=0)
        subprocess.run.return_value = mock_run

        # Test config
        cfg = DictConfig(
            {
                "sweep_job_overrides": {},
                "sim_name": "default",
            }
        )

        # Call with duplicate args
        launch_training_subprocess(
            run_name="test_run",
            protein_suggestion={},
            sweep_name="test_sweep",
            wandb_entity="test_entity",
            wandb_project="test_project",
            cfg=cfg,
            original_args=["run=old_run", "sweep_name=old_sweep", "--gpus=4"],
        )

        # Verify subprocess was called
        call_args = subprocess.run.call_args[0][0]

        # Should have filtered out duplicate run and sweep_name
        assert call_args.count("run=test_run") == 1
        assert "run=old_run" not in call_args
        assert "sweep_name=old_sweep" not in call_args
        assert "--gpus=4" in call_args  # Should keep this
