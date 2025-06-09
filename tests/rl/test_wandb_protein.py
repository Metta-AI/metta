from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from metta.rl.carbs.metta_protein import MettaProtein


class MockConfig:
    """Mock WandB config that behaves like a dictionary."""

    def __init__(self):
        self._data = {}

    def update(self, data, allow_val_change=False):
        self._data.update(data)

    def items(self):
        return self._data.items()

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


@pytest.fixture
def mock_wandb_run():
    """Create a mock WandB run with required attributes."""
    run = MagicMock()
    run.id = "test_run_id"
    run.name = "test_run"
    run.sweep_id = "test_sweep_id"
    run.entity = "test_entity"
    run.project = "test_project"
    run.summary = {}
    run.config = MockConfig()
    return run


@pytest.fixture
def mock_wandb_api():
    """Create a mock WandB API with proper project/run handling."""
    api = MagicMock()

    # Mock the project response
    project = MagicMock()
    project.name = "test_project"
    api.project.return_value = project

    # Mock the runs response
    runs = MagicMock()
    runs.objects = []
    runs.last_response = {"project": project}
    runs.convert_objects = MagicMock(return_value=[])
    api.runs.return_value = runs

    return api


@pytest.fixture
def basic_config():
    """Create a basic sweep configuration."""
    return OmegaConf.create(
        {
            "sweep": {
                "metric": "reward",
                "goal": "maximize",
                "parameters": {
                    "learning_rate": {"min": 1e-5, "max": 1e-1, "mean": 1e-3, "scale": 1, "distribution": "log_normal"}
                },
            }
        }
    )


def test_init_with_wandb(mock_wandb_run, mock_wandb_api):
    """Test initialization with WandB run."""
    with patch("wandb.Api", return_value=mock_wandb_api):
        protein = MettaProtein(
            OmegaConf.create(
                {
                    "sweep": {
                        "metric": "reward",
                        "parameters": {
                            "learning_rate": {
                                "min": 1e-5,
                                "max": 1e-1,
                                "mean": 1e-3,
                                "scale": 1,
                                "distribution": "log_normal",
                            }
                        },
                    }
                }
            ),
            mock_wandb_run,
        )

        # Check WandB state initialization
        assert mock_wandb_run.summary["protein.state"] == "running"
        assert hasattr(protein, "wandb_run")
        assert protein.wandb_run == mock_wandb_run


def test_load_previous_runs(mock_wandb_run, mock_wandb_api):
    """Test loading previous runs from WandB."""
    # Create mock runs with different states
    now = datetime.now(timezone.utc)
    mock_runs = [
        # Initializing run (should be skipped)
        _create_mock_run("run1", {"protein.state": "initializing"}),
        # Running run with recent heartbeat (should be counted)
        _create_mock_run("run2", {"protein.state": "running"}, heartbeat_at=now),
        # Running run with old heartbeat (should be marked defunct)
        _create_mock_run("run3", {"protein.state": "running"}, heartbeat_at=now - timedelta(minutes=10)),
        # Successful run with valid data
        _create_mock_run(
            "run4",
            {"protein.state": "success", "protein.objective": 0.8, "protein.cost": 1.0},
            config={"parameters": {"learning_rate": 0.001}},
            run_id="run4",
        ),
        # Failed run
        _create_mock_run(
            "run5",
            {"protein.state": "failure", "protein.objective": 0.0, "protein.cost": 1.5},
            config={"parameters": {"learning_rate": 0.002}},
        ),
    ]

    # Make the runs object directly iterable
    mock_runs_collection = MagicMock()
    mock_runs_collection.__iter__ = lambda self: iter(mock_runs)
    mock_wandb_api.runs.return_value = mock_runs_collection

    with patch("wandb.Api", return_value=mock_wandb_api):
        protein = MettaProtein(
            OmegaConf.create(
                {
                    "sweep": {
                        "metric": "reward",
                        "parameters": {
                            "learning_rate": {
                                "min": 1e-5,
                                "max": 1e-1,
                                "mean": 1e-3,
                                "scale": 1,
                                "distribution": "log_normal",
                            }
                        },
                    }
                }
            ),
            mock_wandb_run,
        )

        # Check counters
        assert protein._num_observations == 1  # One successful run
        assert protein._num_failures == 1  # One failed run
        assert protein._num_running == 1  # One valid running run
        assert protein._defunct == 1  # One defunct run
        assert protein._invalid == 0  # No invalid runs


def test_record_observation(mock_wandb_run, mock_wandb_api):
    """Test recording observations to WandB."""
    with patch("wandb.Api", return_value=mock_wandb_api):
        protein = MettaProtein(
            OmegaConf.create(
                {
                    "sweep": {
                        "metric": "reward",
                        "parameters": {
                            "learning_rate": {
                                "min": 1e-5,
                                "max": 1e-1,
                                "mean": 1e-3,
                                "scale": 1,
                                "distribution": "log_normal",
                            }
                        },
                    }
                }
            ),
            mock_wandb_run,
        )

        # Record an observation
        MettaProtein._record_observation(mock_wandb_run, 0.5, 1.0)

        # Check WandB updates
        assert mock_wandb_run.summary["protein.state"] == "success"
        assert mock_wandb_run.summary["protein.objective"] == 0.5
        assert mock_wandb_run.summary["protein.cost"] == 1.0


def test_record_failure(mock_wandb_run, mock_wandb_api):
    """Test recording failures to WandB."""
    with patch("wandb.Api", return_value=mock_wandb_api):
        protein = MettaProtein(
            OmegaConf.create(
                {
                    "sweep": {
                        "metric": "reward",
                        "parameters": {
                            "learning_rate": {
                                "min": 1e-5,
                                "max": 1e-1,
                                "mean": 1e-3,
                                "scale": 1,
                                "distribution": "log_normal",
                            }
                        },
                    }
                }
            ),
            mock_wandb_run,
        )

        # Record a failure
        protein.record_failure()

        # Check WandB updates
        assert mock_wandb_run.summary["protein.state"] == "failure"


def test_suggest_with_wandb(mock_wandb_run, mock_wandb_api, basic_config):
    """Test suggestion generation with WandB integration."""
    with patch("wandb.Api", return_value=mock_wandb_api):
        protein = MettaProtein(basic_config, mock_wandb_run)

        # Get a suggestion
        suggestion, _ = protein.suggest()

        # Check suggestion format
        assert "suggestion_uuid" in suggestion
        assert suggestion["suggestion_uuid"] == mock_wandb_run.id
        assert "learning_rate" in suggestion


def test_invalid_run_handling(mock_wandb_run, mock_wandb_api):
    """Test handling of invalid runs."""
    # Create a run that will raise an exception when processed
    bad_run = _create_mock_run(
        "bad_run",
        {"protein.state": "success", "protein.objective": 0.8, "protein.cost": 1.0},
        run_id="bad_run",
    )

    # Create a config that will raise an exception when accessing items
    class BadConfig:
        def items(self):
            raise RuntimeError("Config access error")

    bad_run.config = BadConfig()
    bad_run.id = "bad_run"

    # Make the runs object directly iterable
    mock_runs_collection = MagicMock()
    mock_runs_collection.__iter__ = lambda self: iter([bad_run])
    mock_wandb_api.runs.return_value = mock_runs_collection

    with patch("wandb.Api", return_value=mock_wandb_api):
        protein = MettaProtein(
            OmegaConf.create(
                {
                    "sweep": {
                        "metric": "reward",
                        "parameters": {
                            "learning_rate": {
                                "min": 1e-5,
                                "max": 1e-1,
                                "mean": 1e-3,
                                "scale": 1,
                                "distribution": "log_normal",
                            }
                        },
                    }
                }
            ),
            mock_wandb_run,
        )

        # Check that the run was marked as invalid
        assert protein._invalid == 1
        assert protein._num_observations == 0


def test_observation_state_validation(mock_wandb_run, mock_wandb_api):
    """Test validation of run state when recording observations."""
    with patch("wandb.Api", return_value=mock_wandb_api):
        protein = MettaProtein(
            OmegaConf.create(
                {
                    "sweep": {
                        "metric": "reward",
                        "parameters": {
                            "learning_rate": {
                                "min": 1e-5,
                                "max": 1e-1,
                                "mean": 1e-3,
                                "scale": 1,
                                "distribution": "log_normal",
                            }
                        },
                    }
                }
            ),
            mock_wandb_run,
        )

        # Change state to something other than running
        mock_wandb_run.summary["protein.state"] = "failure"

        # Attempt to record observation should raise an error
        with pytest.raises(AssertionError):
            MettaProtein._record_observation(mock_wandb_run, 0.5, 1.0)

        # Should work with allow_update=True
        MettaProtein._record_observation(mock_wandb_run, 0.5, 1.0, allow_update=True)


def _create_mock_run(name, summary, heartbeat_at=None, config=None, run_id=None):
    """Helper to create mock runs with specific attributes."""
    run = MagicMock()
    run.name = name
    run.id = run_id or name
    run.summary = summary

    # Create a proper config mock
    if config is not None:
        mock_config = MockConfig()
        mock_config.update(config)
        run.config = mock_config
    else:
        run.config = MockConfig()
        run.config.update({"parameters": {"learning_rate": 0.001}})

    run._attrs = {"heartbeatAt": heartbeat_at.strftime("%Y-%m-%dT%H:%M:%S%fZ") if heartbeat_at else None}
    return run
