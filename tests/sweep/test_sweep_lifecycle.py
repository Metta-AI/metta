"""Test sweep lifecycle functions."""

from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig

from metta.sweep.sweep_lifecycle import (
    evaluate_rollout,
    prepare_sweep_run,
    setup_sweep,
)


class TestSetupSweep:
    """Test setup_sweep function."""

    @patch("metta.sweep.sweep_lifecycle.os.makedirs")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    def test_setup_sweep_creates_new_sweep(self, mock_cogweb_class, mock_save, mock_makedirs):
        """Test creating a new sweep."""
        # Setup config
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "sweep_dir": "/tmp/sweep",
                "runs_dir": "/tmp/sweep/runs",
                "sweep_server_uri": "http://test-server",
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
        result = setup_sweep(config, logger)

        # Assertions
        assert result == "test_sweep"
        mock_sweep_client.create_sweep.assert_called_once()
        mock_makedirs.assert_called_once_with("/tmp/sweep/runs", exist_ok=True)
        mock_save.assert_called_once()

    @patch("metta.sweep.sweep_lifecycle.os.makedirs")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    def test_setup_sweep_reuses_existing(self, mock_cogweb_class, mock_save, mock_makedirs):
        """Test reusing an existing sweep."""
        # Setup config
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "sweep_dir": "/tmp/sweep",
                "runs_dir": "/tmp/sweep/runs",
                "sweep_server_uri": "http://test-server",
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
        result = setup_sweep(config, logger)

        # Assertions
        assert result == "test_sweep"
        mock_sweep_client.create_sweep.assert_not_called()
        mock_makedirs.assert_not_called()  # No directory creation for existing sweep
        mock_save.assert_called_once()


class TestPrepareSweepRun:
    """Test prepare_sweep_run function."""

    @patch("metta.sweep.sweep_lifecycle.create_wandb_run_for_sweep")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    @patch("metta.sweep.sweep_lifecycle.os.makedirs")
    @patch("metta.sweep.sweep_lifecycle.apply_protein_suggestion")
    @patch("metta.sweep.sweep_lifecycle.generate_protein_suggestion")
    @patch("metta.sweep.sweep_lifecycle.fetch_protein_observations_from_wandb")
    @patch("metta.sweep.sweep_lifecycle.MettaProtein")
    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    def test_prepare_sweep_run_basic(
        self,
        mock_cogweb_class,
        mock_protein_class,
        mock_fetch_obs,
        mock_gen_suggestion,
        mock_apply_suggestion,
        mock_makedirs,
        mock_save,
        mock_create_wandb_run,
    ):
        """Test basic sweep run preparation."""
        # Setup config - minimal required fields
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "sweep_dir": "/tmp/sweep",
                "sweep_server_uri": "http://test-server",
                "data_dir": "/tmp/sweep/runs",
                "max_observations_to_load": 100,
                "wandb": {
                    "entity": "test_entity",
                    "project": "test_project",
                },
                "sweep": {
                    "name": "test_sweep",
                    "metric": "reward",
                },
                "sweep_train_job": {
                    "run": "${run}",
                    "data_dir": "${data_dir}",
                    "run_dir": "${data_dir}/${run}",
                    "wandb": {
                        "enabled": True,
                        "group": "${run}",
                        "name": "${run}",
                    },
                },
                # Top-level fields that get copied
                "trainer": {
                    "batch_size": 2048,
                    "minibatch_size": 64,
                },
                "device": "cpu",
                "seed": 42,
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
        mock_fetch_obs.return_value = []
        mock_gen_suggestion.return_value = {"trainer": {"lr": 0.001}}

        # Mock WandB run creation
        mock_create_wandb_run.return_value = "test_wandb_run_id"

        # Call function
        logger = Mock()
        run_name, train_cfg, suggestion, wandb_run_id = prepare_sweep_run(config, logger)

        # Assertions
        assert run_name == "test_sweep.r.1"
        assert isinstance(train_cfg, DictConfig)
        assert train_cfg.run == "test_sweep.r.1"
        assert train_cfg.wandb.group == "test_sweep"  # Manually set in the function
        assert train_cfg.wandb.name == "test_sweep.r.1"  # Manually set
        assert suggestion == {"trainer": {"lr": 0.001}}
        assert wandb_run_id == "test_wandb_run_id"

        # Verify directories were created
        mock_makedirs.assert_called()
        # Verify protein suggestion was applied
        mock_apply_suggestion.assert_called_once()


class TestEvaluateRollout:
    """Test evaluate_rollout function."""

    @patch("metta.sweep.sweep_lifecycle.OmegaConf.save")
    @patch("metta.sweep.sweep_lifecycle.record_protein_observation_to_wandb")
    @patch("metta.sweep.sweep_lifecycle._evaluate_sweep_run")
    @patch("metta.sweep.sweep_lifecycle.OmegaConf.to_container")
    @patch("metta.sweep.sweep_lifecycle.WandbContext")
    def test_evaluate_rollout_success(
        self,
        mock_wandb_context_class,
        mock_to_container,
        mock_eval_run,
        mock_record_obs,
        mock_save,
    ):
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
        mock_wandb_context = Mock()
        mock_wandb_context.__enter__ = Mock(return_value=mock_wandb_run)
        mock_wandb_context.__exit__ = Mock(return_value=None)
        mock_wandb_context_class.return_value = mock_wandb_context

        # Mock to_container to extract wandb config properly
        mock_to_container.return_value = {
            "wandb": {
                "entity": "test_entity",
                "project": "test_project",
                "group": "test_sweep",
                "name": "test_sweep.r.1",
            }
        }

        # Mock evaluation results
        mock_eval_run.return_value = {
            "reward": 0.85,
            "time.total": 100.0,
            "time.eval": 50.0,
        }

        # Call function
        logger = Mock()
        suggestion = {"trainer": {"lr": 0.001}}
        result = evaluate_rollout(
            train_cfg,
            suggestion,
            metric="reward",
            sweep_name="test_sweep",
            logger=logger,
        )

        # Assertions
        assert result == {"reward": 0.85, "time.total": 100.0, "time.eval": 50.0}
        mock_eval_run.assert_called_once_with(
            mock_wandb_run,
            "reward",
            "test_sweep",
            train_cfg,
        )
        mock_record_obs.assert_called_once()
        mock_save.assert_called_once()

    @patch("metta.sweep.sweep_lifecycle.OmegaConf.to_container")
    @patch("metta.sweep.sweep_lifecycle.WandbContext")
    def test_evaluate_rollout_wandb_failure(
        self,
        mock_wandb_context_class,
        mock_to_container,
    ):
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

        # Mock to_container
        mock_to_container.return_value = {"wandb": {"entity": "test_entity", "project": "test_project"}}

        # Call function and expect error
        logger = Mock()
        with pytest.raises(RuntimeError, match="WandB initialization failed"):
            evaluate_rollout(
                train_cfg,
                {},
                metric="reward",
                sweep_name="test_sweep",
                logger=logger,
            )
