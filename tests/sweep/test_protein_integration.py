"""Integration tests for the protein sweep pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein


class TestProteinSweepIntegration:
    """Integration tests for the full protein sweep pipeline."""

    @pytest.fixture
    def sweep_dir(self):
        """Create a temporary directory for sweep data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sweep_config(self):
        """Create a realistic sweep configuration."""
        return {
            "sweep_name": "test_integration_sweep",
            "wandb": {"entity": "test_entity", "project": "test_project"},
            "parameters": {
                "trainer": {
                    "optimizer": {
                        "learning_rate": {"min": 0.0001, "max": 0.01},
                        "weight_decay": {"min": 0.0, "max": 0.1},
                    },
                    "batch_size": {"values": [16, 32, 64, 128]},
                    "gradient_clip": {"min": 0.1, "max": 10.0},
                }
            },
            "method": "bayes",
            "metric": "eval/mean_score",
            "goal": "maximize",
            "protein": {
                "search_center": {
                    "trainer/optimizer/learning_rate": 0.001,
                    "trainer/optimizer/weight_decay": 0.01,
                    "trainer/batch_size": 32,
                    "trainer/gradient_clip": 1.0,
                },
                "search_radius": {
                    "trainer/optimizer/learning_rate": 0.5,
                    "trainer/optimizer/weight_decay": 0.5,
                    "trainer/batch_size": 0.5,
                    "trainer/gradient_clip": 0.5,
                },
                "kernel": "matern",
                "gamma": 0.25,
                "xi": 0.001,
            },
        }

    @pytest.fixture
    def mock_wandb_run_factory(self):
        """Factory to create mock wandb runs with different states."""

        def create_run(run_id, name, state="success", objective=None, cost=None, suggestion=None, heartbeat_offset=0):
            run = Mock()
            run.id = run_id
            run.name = name
            run.entity = "test_entity"
            run.project = "test_project"
            run.sweep_id = "test_sweep_123"

            # Mock summary based on state
            summary = {}
            if state != "none":
                summary["protein.state"] = state

            if state == "success" and objective is not None:
                summary["protein.objective"] = objective
                summary["protein.cost"] = cost or 100.0

            if suggestion:
                summary["protein.suggestion"] = suggestion
                summary["protein.suggestion_info"] = {"cost": cost or 100.0}

            if state == "failure":
                summary["protein.error"] = "Test failure"

            run.summary = summary
            run.config = dict(suggestion) if suggestion else {}

            # Mock heartbeat for running states
            if state == "running":
                from datetime import datetime, timedelta, timezone

                if heartbeat_offset == 0:
                    # Recent heartbeat
                    heartbeat = datetime.now(timezone.utc) - timedelta(seconds=30)
                else:
                    # Old heartbeat (defunct)
                    heartbeat = datetime.now(timezone.utc) - timedelta(minutes=10)
                run._attrs = {"heartbeatAt": heartbeat.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}

            return run

        return create_run

    @patch("wandb.Api")
    @patch("wandb.run")
    def test_protein_loads_previous_observations(self, mock_wandb_run, mock_api, mock_wandb_run_factory, sweep_config):
        """Test that protein properly loads and learns from previous runs."""
        # Create current run
        current_run = mock_wandb_run_factory("current_run", "current", state="none")
        current_run.summary = Mock()
        current_run.summary.get.return_value = None
        current_run.summary.update = Mock()
        # Create config mock separately to avoid attribute issues
        config = Mock()
        config._locked = {}
        config.update = Mock()
        current_run.config = config
        mock_wandb_run.return_value = current_run

        # Create historical runs with various states
        historical_runs = [
            # Successful runs
            mock_wandb_run_factory(
                "run1",
                "successful_run_1",
                "success",
                objective=0.85,
                cost=120.0,
                suggestion={"trainer": {"optimizer": {"learning_rate": 0.002}}},
            ),
            mock_wandb_run_factory(
                "run2",
                "successful_run_2",
                "success",
                objective=0.92,
                cost=150.0,
                suggestion={"trainer": {"optimizer": {"learning_rate": 0.005}}},
            ),
            # Failed run
            mock_wandb_run_factory(
                "run3", "failed_run", "failure", suggestion={"trainer": {"optimizer": {"learning_rate": 0.01}}}
            ),
            # Running run (recent heartbeat)
            mock_wandb_run_factory(
                "run4", "running_run", "running", suggestion={"trainer": {"optimizer": {"learning_rate": 0.003}}}
            ),
            # Defunct run (old heartbeat)
            mock_wandb_run_factory(
                "run5",
                "defunct_run",
                "running",
                suggestion={"trainer": {"optimizer": {"learning_rate": 0.004}}},
                heartbeat_offset=1,  # Old heartbeat
            ),
            # Initializing run (should be skipped)
            mock_wandb_run_factory("run6", "initializing_run", "initializing"),
        ]

        # Mock API to return historical runs
        mock_api.return_value.runs.return_value = historical_runs

        # Create MettaProtein
        cfg = OmegaConf.create(sweep_config)
        with patch("metta.sweep.protein_metta.Protein") as mock_protein_class:
            # Mock the Protein instance
            mock_protein = Mock()
            mock_protein.observe = Mock()
            mock_protein.suggest.return_value = (
                {"trainer/optimizer/learning_rate": 0.0035},  # New suggestion
                {"predicted_objective": 0.88, "uncertainty": 0.05},
            )
            mock_protein_class.return_value = mock_protein

            # Create MettaProtein
            MettaProtein(cfg, current_run)

            # Verify observations were recorded
            assert mock_protein.observe.call_count == 3  # 2 successful + 1 failed

            # Check successful observations
            calls = mock_protein.observe.call_args_list

            # First successful run
            assert calls[0][0][0] == {"trainer/optimizer/learning_rate": 0.002}
            assert calls[0][0][1] == 0.85  # objective
            assert calls[0][0][2] == 120.0  # cost
            assert calls[0][0][3] is False  # not a failure

            # Second successful run
            assert calls[1][0][0] == {"trainer/optimizer/learning_rate": 0.005}
            assert calls[1][0][1] == 0.92
            assert calls[1][0][2] == 150.0
            assert calls[1][0][3] is False

            # Failed run
            assert calls[2][0][0] == {"trainer/optimizer/learning_rate": 0.01}
            assert calls[2][0][1] == 0.0  # Failed runs get 0 objective
            assert calls[2][0][2] == 0.0  # Failed runs get 0 cost
            assert calls[2][0][3] is True  # is a failure

    @patch("wandb.Api")
    def test_record_observation_updates_protein(self, mock_api, sweep_config):
        """Test that recording observations updates both WandB and Protein."""
        # Create mock run
        mock_run = Mock()
        mock_run.sweep_id = "test_sweep"
        mock_run.entity = "test_entity"
        mock_run.project = "test_project"
        mock_run.id = "test_run"
        mock_run.name = "test"
        mock_run.summary = Mock()
        mock_run.summary.get.return_value = None
        mock_run.summary.update = Mock()

        # Create config mock separately to avoid attribute issues
        config = Mock()
        config._locked = {}
        config.update = Mock()
        mock_run.config = config

        # Mock API
        mock_api.return_value.runs.return_value = []

        # Create protein with mock
        with patch("metta.sweep.protein_metta.Protein") as mock_protein_class:
            mock_protein = Mock()
            mock_protein.observe = Mock()
            mock_protein.suggest.return_value = ({"trainer/optimizer/learning_rate": 0.003}, {"cost": 100.0})
            mock_protein_class.return_value = mock_protein

            # Create MettaProtein
            cfg = OmegaConf.create(sweep_config)
            metta_protein = MettaProtein(cfg, mock_run)

            # Record an observation
            metta_protein.record_observation(objective=0.95, cost=200.0)

            # Verify WandB was updated
            update_calls = [call[0][0] for call in mock_run.summary.update.call_args_list]

            # Find the observation update
            obs_update = None
            for update in update_calls:
                if "protein.objective" in update:
                    obs_update = update
                    break

            assert obs_update is not None
            assert obs_update["protein.objective"] == 0.95
            assert obs_update["protein.cost"] == 200.0
            assert obs_update["protein.state"] == "success"

            # Verify Protein was updated
            mock_protein.observe.assert_called_once_with({"trainer/optimizer/learning_rate": 0.003}, 0.95, 200.0, False)

    @patch("wandb.Api")
    def test_serialization_in_real_scenario(self, mock_api, sweep_config):
        """Test serialization with realistic numpy and WandB objects."""
        # Create mock run with WandB-like summary
        mock_run = Mock()
        mock_run.sweep_id = "test_sweep"
        mock_run.entity = "test_entity"
        mock_run.project = "test_project"
        mock_run.id = "test_run"
        mock_run.name = "test"

        # Create a mock summary that behaves like WandB's
        class MockWandbSummary:
            def __init__(self):
                self._items = {}

            def get(self, key, default=None):
                return self._items.get(key, default)

            def update(self, updates):
                # Simulate WandB's behavior of storing as special dict
                for k, v in updates.items():
                    self._items[k] = v

            def __getitem__(self, key):
                return self._items[key]

        mock_run.summary = MockWandbSummary()

        # Create config mock separately to avoid attribute issues
        config = Mock()
        config._locked = {}
        config.update = Mock()
        mock_run.config = config

        # Mock API to return runs with numpy types
        historical_run = Mock()
        historical_run.id = "historical"
        historical_run.name = "historical"
        historical_run.summary = {
            "protein.state": "success",
            "protein.objective": np.float64(0.88),
            "protein.cost": np.float32(125.5),
            "protein.suggestion": {
                "trainer": {
                    "optimizer": {"learning_rate": np.float32(0.0045), "weight_decay": np.float64(0.02)},
                    "batch_size": np.int32(64),
                }
            },
            "protein.suggestion_info": {
                "predicted_score": np.float64(0.87),
                "uncertainty": np.float32(0.03),
                "acquisition_value": np.float64(1.25),
            },
        }

        mock_api.return_value.runs.return_value = [historical_run]

        # Create MettaProtein with numpy returns from Protein
        with patch("metta.sweep.protein_metta.Protein") as mock_protein_class:
            mock_protein = Mock()
            mock_protein.observe = Mock()

            # Protein returns numpy types
            mock_protein.suggest.return_value = (
                {
                    "trainer/optimizer/learning_rate": np.float32(0.0038),
                    "trainer/optimizer/weight_decay": np.float64(0.015),
                    "trainer/batch_size": np.int64(32),
                    "trainer/gradient_clip": np.float32(1.5),
                },
                {
                    "predicted_objective": np.float64(0.91),
                    "uncertainty": np.float32(0.02),
                    "ei_value": np.float64(0.95),
                    "iteration": np.int32(5),
                },
            )
            mock_protein_class.return_value = mock_protein

            # Create MettaProtein
            cfg = OmegaConf.create(sweep_config)
            MettaProtein(cfg, mock_run)

            # Get the saved suggestion from wandb summary
            saved_suggestion = mock_run.summary.get("protein.suggestion")
            saved_info = mock_run.summary.get("protein.suggestion_info")

            # Verify everything is JSON serializable
            json.dumps(saved_suggestion)
            json.dumps(saved_info)

            # Verify types were properly converted
            assert isinstance(saved_suggestion["trainer/optimizer/learning_rate"], float)
            assert isinstance(saved_suggestion["trainer/optimizer/weight_decay"], float)
            assert isinstance(saved_suggestion["trainer/batch_size"], int)
            assert isinstance(saved_info["predicted_objective"], float)
            assert isinstance(saved_info["iteration"], int)

            # Verify values are correct
            assert saved_suggestion["trainer/optimizer/learning_rate"] == pytest.approx(0.0038)
            assert saved_suggestion["trainer/batch_size"] == 32
            assert saved_info["predicted_objective"] == pytest.approx(0.91)

    def test_sweep_state_file_compatibility(self, sweep_dir, sweep_config):
        """Test that sweep state files are properly formatted."""
        # Create a state file that might be saved by the sweep
        state_file = sweep_dir / "protein_state.json"

        # Simulate saving state with numpy types
        state_data = {
            "suggestion": {"trainer/optimizer/learning_rate": np.float32(0.004), "trainer/batch_size": np.int64(64)},
            "info": {"cost": np.float64(150.0), "score": np.float32(0.89)},
            "observations": [
                {
                    "suggestion": {"trainer/optimizer/learning_rate": np.float32(0.002)},
                    "objective": np.float64(0.85),
                    "cost": np.float32(100.0),
                    "is_failure": False,
                }
            ],
        }

        # This should fail without proper cleaning
        with pytest.raises(TypeError):
            with open(state_file, "w") as f:
                json.dump(state_data, f)

        # Now clean it using the same logic as WandbProtein
        from metta.common.util.numpy_helpers import clean_numpy_types

        cleaned_state = clean_numpy_types(state_data)

        # This should succeed
        with open(state_file, "w") as f:
            json.dump(cleaned_state, f)

        # Verify we can read it back
        with open(state_file, "r") as f:
            loaded_state = json.load(f)

        assert loaded_state["suggestion"]["trainer/optimizer/learning_rate"] == pytest.approx(0.004)
        assert loaded_state["observations"][0]["objective"] == pytest.approx(0.85)
