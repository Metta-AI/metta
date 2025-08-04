"""Test sweep lifecycle functions."""

from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig

# Skip all tests in this file due to torch import issues on macOS
pytest.skip("Skipping sweep lifecycle tests due to torch import issues", allow_module_level=True)

from metta.sweep.sweep_lifecycle import (  # noqa: E402
    evaluate_sweep_rollout,
    initialize_sweep,
    prepare_sweep_run,
)


class TestInitializeSweep:
    """Test initialize_sweep function."""

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    def test_initialize_new_sweep(self, mock_cogweb_class):
        """Test creating a new sweep."""
        # Setup config
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "settings": {
                    "sweep_server_uri": "http://test-server",
                },
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                },
            }
        )

        # Mock CogwebClient
        mock_client = Mock()
        mock_cogweb_class.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client

        # Mock sweep doesn't exist
        mock_sweep_info = Mock()
        mock_sweep_info.exists = False
        mock_sweep_client.get_sweep.return_value = mock_sweep_info

        # Call function
        logger = Mock()
        initialize_sweep(config, logger)

        # Assertions
        mock_sweep_client.create_sweep.assert_called_once_with(
            "test_sweep", "test_project", "test_entity", "test_sweep"
        )
        logger.info.assert_called()

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    def test_initialize_existing_sweep(self, mock_cogweb_class):
        """Test handling an existing sweep."""
        # Setup config
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "settings": {
                    "sweep_server_uri": "http://test-server",
                },
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                },
            }
        )

        # Mock CogwebClient
        mock_client = Mock()
        mock_cogweb_class.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client

        # Mock sweep exists
        mock_sweep_info = Mock()
        mock_sweep_info.exists = True
        mock_sweep_client.get_sweep.return_value = mock_sweep_info

        # Call function
        logger = Mock()
        initialize_sweep(config, logger)

        # Assertions - should not create sweep
        mock_sweep_client.create_sweep.assert_not_called()
        logger.info.assert_called()


class TestPrepareSweepRun:
    """Test prepare_sweep_run function."""

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    @patch("metta.sweep.sweep_lifecycle.MettaProtein")
    @patch("metta.sweep.sweep_lifecycle.fetch_protein_observations_from_wandb")
    def test_prepare_sweep_run_no_history(self, mock_fetch_obs, mock_protein_class, mock_cogweb_class):
        """Test preparing a sweep run with no previous observations."""
        # Setup config
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "settings": {
                    "sweep_server_uri": "http://test-server",
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
            }
        )

        # Mock CogwebClient
        mock_client = Mock()
        mock_cogweb_class.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client
        mock_sweep_client.get_next_run_id.return_value = "test_sweep.r.1"

        # Mock Protein
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein
        mock_protein.suggest.return_value = ({"trainer": {"lr": 0.001}}, None)

        # Mock fetch observations
        mock_fetch_obs.return_value = []

        # Call function
        logger = Mock()
        run_name, suggestion = prepare_sweep_run(config, logger)

        # Assertions
        assert run_name == "test_sweep.r.1"
        assert suggestion == {"trainer": {"lr": 0.001}}
        mock_protein.observe.assert_not_called()
        mock_protein.suggest.assert_called_once()

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    @patch("metta.sweep.sweep_lifecycle.MettaProtein")
    @patch("metta.sweep.sweep_lifecycle.fetch_protein_observations_from_wandb")
    def test_prepare_sweep_run_with_history(self, mock_fetch_obs, mock_protein_class, mock_cogweb_class):
        """Test preparing a sweep run with previous observations."""
        # Setup config
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "settings": {
                    "sweep_server_uri": "http://test-server",
                    "max_observations_to_load": 100,
                },
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                },
                "sweep": {
                    "metric": "reward",
                },
            }
        )

        # Mock CogwebClient
        mock_client = Mock()
        mock_cogweb_class.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client
        mock_sweep_client.get_next_run_id.return_value = "test_sweep.r.5"

        # Mock Protein
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein
        mock_protein.suggest.return_value = ({"trainer": {"lr": 0.005}}, None)

        # Mock fetch observations with history
        mock_fetch_obs.return_value = [
            {
                "suggestion": {"trainer": {"lr": 0.001}},
                "objective": 0.8,
                "cost": 100,
                "is_failure": False,
            },
            {
                "suggestion": {"trainer": {"lr": 0.002}},
                "objective": 0.85,
                "cost": 120,
                "is_failure": False,
            },
        ]

        # Call function
        logger = Mock()
        run_name, suggestion = prepare_sweep_run(config, logger)

        # Assertions
        assert run_name == "test_sweep.r.5"
        assert suggestion == {"trainer": {"lr": 0.005}}
        assert mock_protein.observe.call_count == 2
        mock_protein.suggest.assert_called_once()


class TestEvaluateSweepRollout:
    """Test evaluate_sweep_rollout function."""

    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    @patch("metta.sweep.sweep_lifecycle.record_protein_observation_to_wandb")
    @patch("metta.sweep.sweep_lifecycle._run_policy_evaluation")
    @patch("metta.sweep.sweep_lifecycle.WandbContext")
    def test_evaluate_sweep_rollout_success(self, mock_wandb_context_class, mock_eval_run, mock_record_obs, mock_save):
        """Test successful rollout evaluation."""
        # Setup config
        train_cfg = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/sweep/runs/test_sweep.r.1",
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                    "group": "test_sweep",
                    "name": "test_sweep.r.1",
                },
            }
        )

        # Mock WandB context
        mock_wandb_run = Mock()
        mock_wandb_run.summary = Mock()
        mock_wandb_context = Mock()
        mock_wandb_context.__enter__ = Mock(return_value=mock_wandb_run)
        mock_wandb_context.__exit__ = Mock(return_value=None)
        mock_wandb_context_class.return_value = mock_wandb_context

        # Mock evaluation results
        mock_eval_run.return_value = {
            "reward": 0.85,
            "time.total": 100.0,
            "time.eval": 50.0,
            "time.train": 50.0,
            "cost.total": 1.5,
            "cost.training": 1.0,
            "cost.eval": 0.5,
        }

        # Call function
        suggestion = {"trainer": {"lr": 0.001}}
        result = evaluate_sweep_rollout(
            train_cfg,
            suggestion,
            metric="reward",
            sweep_name="test_sweep",
        )

        # Assertions
        assert result["reward"] == 0.85
        assert result["time.total"] == 100.0
        mock_eval_run.assert_called_once_with(
            mock_wandb_run,
            "reward",
            "test_sweep",
            train_cfg,
        )
        mock_record_obs.assert_called_once_with(
            mock_wandb_run,
            suggestion,
            0.85,  # score
            100.0,  # cost (time.total)
            False,  # is_failure
        )
        mock_save.assert_called_once()
        mock_wandb_run.summary.update.assert_called_once()

    @patch("metta.sweep.sweep_lifecycle.WandbContext")
    def test_evaluate_sweep_rollout_wandb_failure(self, mock_wandb_context_class):
        """Test error when WandB fails to initialize."""
        # Setup config
        train_cfg = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/sweep/runs/test_sweep.r.1",
                "wandb": {"entity": "test_entity", "project": "test_project"},
            }
        )

        # Mock WandB context to return None
        mock_wandb_context = Mock()
        mock_wandb_context.__enter__ = Mock(return_value=None)
        mock_wandb_context.__exit__ = Mock(return_value=None)
        mock_wandb_context_class.return_value = mock_wandb_context

        # Call function and expect error
        with pytest.raises(RuntimeError, match="WandB initialization failed"):
            evaluate_sweep_rollout(
                train_cfg,
                {},
                metric="reward",
                sweep_name="test_sweep",
            )

    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    @patch("metta.sweep.sweep_lifecycle.record_protein_observation_to_wandb")
    @patch("metta.sweep.sweep_lifecycle._run_policy_evaluation")
    @patch("metta.sweep.sweep_lifecycle.WandbContext")
    @patch("metta.sweep.sweep_lifecycle.os.environ.get")
    def test_evaluate_sweep_rollout_with_hourly_cost(
        self,
        mock_environ_get,
        mock_wandb_context_class,
        mock_eval_run,
        mock_record_obs,
        mock_save,
    ):
        """Test evaluation with hourly cost from environment."""
        # Setup config
        train_cfg = DictConfig(
            {
                "run": "test_sweep.r.1",
                "run_dir": "/tmp/sweep/runs/test_sweep.r.1",
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                },
            }
        )

        # Mock environment variable
        mock_environ_get.return_value = "4.5"

        # Mock WandB context
        mock_wandb_run = Mock()
        mock_wandb_run.summary = Mock()
        mock_wandb_context = Mock()
        mock_wandb_context.__enter__ = Mock(return_value=mock_wandb_run)
        mock_wandb_context.__exit__ = Mock(return_value=None)
        mock_wandb_context_class.return_value = mock_wandb_context

        # Mock evaluation results (will be calculated with hourly cost)
        mock_eval_run.return_value = {
            "reward": 0.9,
            "time.total": 3600.0,  # 1 hour
            "time.eval": 1800.0,
            "time.train": 1800.0,
            "cost.hourly": 4.5,
            "cost.training": 2.25,
            "cost.eval": 2.25,
            "cost.total": 4.5,
        }

        # Call function
        result = evaluate_sweep_rollout(
            train_cfg,
            {"trainer": {"lr": 0.001}},
            metric="reward",
            sweep_name="test_sweep",
        )

        # Assertions
        assert result["cost.hourly"] == 4.5
        assert result["cost.total"] == 4.5
