"""Unit tests for sweep lifecycle functions."""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest
from omegaconf import DictConfig

from metta.sweep.sweep_lifecycle import (
    evaluate_rollout,
    evaluate_sweep_run,
    prepare_sweep_run,
    setup_sweep,
)


class TestSetupSweep:
    """Test the setup_sweep function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return DictConfig(
            {
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
            }
        )

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    @patch("metta.sweep.sweep_lifecycle.create_wandb_sweep")
    @patch("metta.sweep.sweep_lifecycle.os.makedirs")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    def test_setup_sweep_creates_new_sweep(
        self, mock_save, mock_makedirs, mock_create_wandb, mock_cogweb_client, mock_config, mock_logger
    ):
        """Test creating a new sweep when it doesn't exist."""
        # Setup mocks
        mock_client = Mock()
        mock_cogweb_client.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client

        # Sweep doesn't exist
        mock_sweep_info = Mock()
        mock_sweep_info.exists = False
        mock_sweep_info.wandb_sweep_id = None
        mock_sweep_client.get_sweep.return_value = mock_sweep_info

        # WandB returns new sweep ID
        mock_create_wandb.return_value = "new_wandb_sweep_123"

        # Call function
        result = setup_sweep(mock_config, mock_logger)

        # Assertions
        assert result == "new_wandb_sweep_123"
        mock_create_wandb.assert_called_once_with("test_entity", "test_project", "test_sweep")
        mock_sweep_client.create_sweep.assert_called_once_with(
            "test_sweep", "test_project", "test_entity", "new_wandb_sweep_123"
        )
        assert mock_save.call_count == 2  # Called twice for metadata
        mock_logger.info.assert_any_call("Creating sweep test_sweep in WandB")

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    @patch("metta.sweep.sweep_lifecycle.create_wandb_sweep")
    @patch("metta.sweep.sweep_lifecycle.os.makedirs")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    def test_setup_sweep_reuses_existing_sweep(
        self, mock_save, mock_makedirs, mock_create_wandb, mock_cogweb_client, mock_config, mock_logger
    ):
        """Test reusing an existing sweep."""
        # Setup mocks
        mock_client = Mock()
        mock_cogweb_client.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client

        # Sweep already exists
        mock_sweep_info = Mock()
        mock_sweep_info.exists = True
        mock_sweep_info.wandb_sweep_id = "existing_wandb_sweep_456"
        mock_sweep_client.get_sweep.return_value = mock_sweep_info

        # Call function
        result = setup_sweep(mock_config, mock_logger)

        # Assertions
        assert result == "existing_wandb_sweep_456"
        mock_create_wandb.assert_not_called()  # Should not create new sweep
        mock_sweep_client.create_sweep.assert_not_called()
        assert mock_save.call_count == 1  # Only saves metadata once
        mock_logger.info.assert_any_call(
            "Found existing sweep test_sweep in the centralized DB. WandB sweep ID: existing_wandb_sweep_456"
        )

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    def test_setup_sweep_handles_cogweb_errors(self, mock_cogweb_client, mock_config, mock_logger):
        """Test error handling when Cogweb fails."""
        # Setup mock to raise exception
        mock_cogweb_client.get_client.side_effect = Exception("Connection failed")

        # Call function and expect exception
        with pytest.raises(Exception, match="Connection failed"):
            setup_sweep(mock_config, mock_logger)


class TestPrepareSweepRun:
    """Test the prepare_sweep_run function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return DictConfig(
            {
                "sweep": {
                    "metric": "reward",
                    "parameters": {"learning_rate": {"min": 0.001, "max": 0.01}},
                },
                "trainer": {
                    "batch_size": 2048,
                    "minibatch_size": 64,
                    "bptt_horizon": 32,
                },
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                },
                "sweep_id": "test_sweep_123",
                "sweep_name": "test_sweep",
                "sweep_server_uri": "http://test-server",
                "data_dir": "/tmp/data",
                "runs_dir": "/tmp/test_sweep/runs",
                "max_observations_to_load": 100,
                "sweep_train_job": {
                    "trainer": {"learning_rate": 0.001},
                },
            }
        )

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)

    @patch("metta.sweep.sweep_lifecycle.MettaProtein")
    @patch("metta.sweep.sweep_lifecycle.fetch_protein_observations_from_wandb")
    @patch("metta.sweep.sweep_lifecycle.generate_protein_suggestion")
    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    @patch("metta.sweep.sweep_lifecycle.create_wandb_run_for_sweep")
    @patch("metta.sweep.sweep_lifecycle.os.makedirs")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    def test_prepare_sweep_run_generates_valid_config(
        self,
        mock_save,
        mock_makedirs,
        mock_create_wandb_run,
        mock_cogweb_client,
        mock_generate_suggestion,
        mock_fetch_observations,
        mock_protein_class,
        mock_config,
        mock_logger,
    ):
        """Test successful sweep run preparation."""
        # Setup mocks
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein

        mock_fetch_observations.return_value = [
            {"suggestion": {"lr": 0.002}, "objective": 0.8, "cost": 100, "is_failure": False}
        ]

        mock_generate_suggestion.return_value = {"trainer": {"learning_rate": 0.005}}

        mock_client = Mock()
        mock_cogweb_client.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client
        mock_sweep_client.get_next_run_id.return_value = "test_sweep.r.42"

        # Call function
        run_name, downstream_cfg, protein_suggestion = prepare_sweep_run(mock_config, mock_logger)

        # Assertions
        assert run_name == "test_sweep.r.42"
        assert isinstance(downstream_cfg, DictConfig)
        assert downstream_cfg.run == "test_sweep.r.42"
        assert downstream_cfg.data_dir == "/tmp/data/sweep/test_sweep/runs"
        assert protein_suggestion == {"trainer": {"learning_rate": 0.005}}

        mock_protein.observe.assert_called_once()
        mock_create_wandb_run.assert_called_once()
        mock_save.assert_called_once()  # For train_config_overrides.yaml

    @patch("metta.sweep.sweep_lifecycle.MettaProtein")
    @patch("metta.sweep.sweep_lifecycle.fetch_protein_observations_from_wandb")
    @patch("metta.sweep.sweep_lifecycle.generate_protein_suggestion")
    @patch("metta.sweep.sweep_lifecycle.apply_protein_suggestion")
    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    @patch("metta.sweep.sweep_lifecycle.create_wandb_run_for_sweep")
    @patch("metta.sweep.sweep_lifecycle.os.makedirs")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    def test_prepare_sweep_run_applies_protein_suggestion(
        self,
        mock_save,
        mock_makedirs,
        mock_create_wandb_run,
        mock_cogweb_client,
        mock_apply_suggestion,
        mock_generate_suggestion,
        mock_fetch_observations,
        mock_protein_class,
        mock_config,
        mock_logger,
    ):
        """Test that protein suggestions are properly applied to config."""
        # Setup mocks
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein
        mock_fetch_observations.return_value = []

        suggestion = {"trainer": {"batch_size": 4096}}
        mock_generate_suggestion.return_value = suggestion

        mock_client = Mock()
        mock_cogweb_client.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client
        mock_sweep_client.get_next_run_id.return_value = "test_sweep.r.1"

        # Call function
        prepare_sweep_run(mock_config, mock_logger)

        # Verify apply_protein_suggestion was called
        mock_apply_suggestion.assert_called_once()
        call_args = mock_apply_suggestion.call_args[0]
        assert call_args[0].run == "test_sweep.r.1"  # sweep_job_cfg
        assert call_args[1] == suggestion


class TestEvaluateRollout:
    """Test the evaluate_rollout function."""

    @pytest.fixture
    def mock_downstream_config(self):
        """Create a mock downstream configuration."""
        return DictConfig(
            {
                "run": "test_sweep.r.1",
                "data_dir": "/tmp/sweep/test_sweep/runs",
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                },
                "sweep": {
                    "metric": "reward",
                    "name": "test_sweep",
                },
                "sweep_name": "test_sweep",
            }
        )

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)

    @patch("metta.sweep.sweep_lifecycle.WandbContext")
    @patch("metta.sweep.sweep_lifecycle.evaluate_sweep_run")
    @patch("metta.sweep.sweep_lifecycle.record_protein_observation_to_wandb")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    def test_evaluate_rollout_computes_metrics(
        self,
        mock_save,
        mock_record_observation,
        mock_evaluate_sweep,
        mock_wandb_context,
        mock_downstream_config,
        mock_logger,
    ):
        """Test that evaluation computes and returns metrics."""
        # Setup mocks
        mock_wandb_run = MagicMock()
        mock_wandb_context.return_value.__enter__.return_value = mock_wandb_run

        eval_results = {
            "time.total": 150.0,
            "time.eval": 50.0,
            "reward": 0.85,
            "score": 0.85,
        }
        mock_evaluate_sweep.return_value = eval_results

        protein_suggestion = {"trainer": {"learning_rate": 0.005}}

        # Call function
        result = evaluate_rollout(mock_downstream_config, protein_suggestion, mock_logger)

        # Assertions
        assert result == eval_results
        mock_evaluate_sweep.assert_called_once_with(
            mock_wandb_run,
            "reward",
            "test_sweep",
            mock_downstream_config,
        )
        mock_record_observation.assert_called_once_with(
            mock_wandb_run,
            protein_suggestion,
            0.85,  # score
            150.0,  # cost
            False,  # is_failure
        )
        mock_wandb_run.summary.update.assert_called_once_with(eval_results)

    @patch("metta.sweep.sweep_lifecycle.WandbContext")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    def test_evaluate_rollout_saves_results(
        self,
        mock_save,
        mock_wandb_context,
        mock_downstream_config,
        mock_logger,
    ):
        """Test that evaluation results are saved to file."""
        # Setup mocks
        mock_wandb_run = MagicMock()
        mock_wandb_context.return_value.__enter__.return_value = mock_wandb_run

        # Mock evaluate_sweep_run
        with patch("metta.sweep.sweep_lifecycle.evaluate_sweep_run") as mock_evaluate:
            mock_evaluate.return_value = {
                "time.total": 100.0,
                "reward": 0.9,
            }

            # Call function
            evaluate_rollout(mock_downstream_config, {}, mock_logger)

            # Verify save was called with correct path and data
            mock_save.assert_called_once_with(
                {"eval_metric": 0.9, "total_time": 100.0},
                "/tmp/sweep/test_sweep/runs/sweep_eval_results.yaml",
            )

    @patch("metta.sweep.sweep_lifecycle.WandbContext")
    def test_evaluate_rollout_handles_wandb_failure(
        self,
        mock_wandb_context,
        mock_downstream_config,
        mock_logger,
    ):
        """Test error handling when WandB initialization fails."""
        # Setup mock to return None
        mock_wandb_context.return_value.__enter__.return_value = None

        # Call function and expect exception
        with pytest.raises(RuntimeError, match="WandB initialization failed"):
            evaluate_rollout(mock_downstream_config, {}, mock_logger)


class TestEvaluateSweepRun:
    """Test the evaluate_sweep_run function."""

    @pytest.fixture
    def mock_global_config(self):
        """Create a mock global configuration."""
        return DictConfig(
            {
                "sim": {
                    "num_envs": 1,
                    "num_episodes": 10,
                },
                "device": "cpu",
                "vectorization": "none",
            }
        )

    @patch("metta.sweep.sweep_lifecycle.SimulationSuiteConfig")
    @patch("metta.sweep.sweep_lifecycle.PolicyStore")
    @patch("metta.sweep.sweep_lifecycle.SimulationSuite")
    @patch("metta.sweep.sweep_lifecycle.EvalStatsDB")
    def test_evaluate_sweep_run_with_missing_policy(
        self,
        mock_eval_stats_db,
        mock_sim_suite_class,
        mock_policy_store_class,
        mock_sim_config_class,
        mock_global_config,
    ):
        """Test error handling when policy is missing."""
        # Setup mocks
        mock_wandb_run = Mock()
        mock_wandb_run.name = "test_run"

        mock_policy_store = Mock()
        mock_policy_store_class.return_value = mock_policy_store

        # Policy record with no URI
        mock_policy_record = Mock()
        mock_policy_record.uri = None
        mock_policy_store.policy_record.return_value = mock_policy_record

        # Call function and expect exception
        with pytest.raises(ValueError, match="Policy record has no URI"):
            evaluate_sweep_run(
                mock_wandb_run,
                "reward",
                "test_sweep",
                mock_global_config,
            )

    @patch("metta.sweep.sweep_lifecycle.SimulationSuiteConfig")
    @patch("metta.sweep.sweep_lifecycle.PolicyStore")
    @patch("metta.sweep.sweep_lifecycle.SimulationSuite")
    @patch("metta.sweep.sweep_lifecycle.EvalStatsDB")
    @patch("metta.sweep.sweep_lifecycle.time.time")
    def test_evaluate_sweep_run_success(
        self,
        mock_time,
        mock_eval_stats_db_class,
        mock_sim_suite_class,
        mock_policy_store_class,
        mock_sim_config_class,
        mock_global_config,
    ):
        """Test successful sweep run evaluation."""
        # Setup time mocks
        mock_time.side_effect = [100.0, 150.0]  # Start and end time

        # Setup mocks
        mock_wandb_run = Mock()
        mock_wandb_run.name = "test_run"
        mock_wandb_run.summary = Mock(update=Mock())

        mock_policy_store = Mock()
        mock_policy_store_class.return_value = mock_policy_store

        # Policy record
        mock_policy_record = Mock()
        mock_policy_record.uri = "wandb://run/test_run"
        mock_policy_record.metadata = {
            "train_time": 100.0,
            "agent_step": 1000,
            "epoch": 10,
        }
        mock_policy_store.policy_record.return_value = mock_policy_record
        mock_policy_store.load_from_uri.return_value = mock_policy_record

        # Simulation suite
        mock_sim_suite = Mock()
        mock_sim_suite_class.return_value = mock_sim_suite
        mock_results = Mock()
        mock_results.stats_db = Mock()
        mock_sim_suite.simulate.return_value = mock_results

        # Eval stats
        mock_eval_stats_db = Mock()
        mock_eval_stats_db_class.from_sim_stats_db.return_value = mock_eval_stats_db
        mock_eval_stats_db.get_average_metric_by_filter.return_value = 0.95

        # Call function
        result = evaluate_sweep_run(
            mock_wandb_run,
            "reward",
            "test_sweep",
            mock_global_config,
        )

        # Assertions
        assert result["score"] == 0.95
        assert result["time.eval"] == 50.0  # 150 - 100
        assert result["time.total"] == 150.0  # train_time + eval_time
        assert result["train.agent_step"] == 1000

        mock_policy_store.add_to_wandb_sweep.assert_called_once_with("test_sweep", mock_policy_record)
        mock_wandb_run.summary.update.assert_called_once()
