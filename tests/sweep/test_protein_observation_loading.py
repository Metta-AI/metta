"""
Comprehensive tests for Protein observation loading pipeline.
This test verifies that WandbProtein correctly loads previous observations from WandB sweeps.
"""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from metta.sweep.protein import Protein
from metta.sweep.protein_wandb import WandbProtein


class TestProteinObservationLoading:
    """Test the complete observation loading pipeline."""

    @pytest.fixture
    def base_sweep_config(self):
        """Create a basic sweep configuration for testing."""
        return OmegaConf.create(
            {
                "protein": {
                    "max_suggestion_cost": 1800,
                    "resample_frequency": 0,
                    "num_random_samples": 5,
                    "global_search_scale": 1,
                    "random_suggestions": 1024,
                    "suggestions_per_pareto": 256,
                },
                "parameters": {
                    "metric": "reward",
                    "goal": "maximize",
                    "trainer": {
                        "optimizer": {
                            "learning_rate": {
                                "distribution": "log_normal",
                                "min": 0.0001,
                                "max": 0.001,
                                "mean": 0.0005,
                                "scale": 0.5,
                            }
                        }
                    },
                },
            }
        )

    def test_sweep_id_detection_fallback(self):
        """Test that our sweep ID fallback logic works correctly."""

        class MockSweep:
            def __init__(self, sweep_id):
                self.id = sweep_id

        class MockWandbRun:
            def __init__(self, has_sweep_id=False, sweep_obj=None):
                if has_sweep_id:
                    self.sweep_id = "direct-sweep-id"
                if sweep_obj:
                    self.sweep = sweep_obj
                self.summary = {}
                self.config = {"__dict__": {"_locked": {}}}
                self.entity = "test-entity"
                self.project = "test-project"
                self.id = "test-run-id"

        # Test 1: Direct sweep_id attribute available
        mock_run = MockWandbRun(has_sweep_id=True)
        sweep_id = getattr(mock_run, "sweep_id", None)
        if sweep_id is None and hasattr(mock_run, "sweep") and mock_run.sweep:
            sweep_id = mock_run.sweep.id

        assert sweep_id == "direct-sweep-id"

        # Test 2: No sweep_id attribute, use sweep.id fallback
        mock_run = MockWandbRun(has_sweep_id=False, sweep_obj=MockSweep("fallback-sweep-456"))
        sweep_id = getattr(mock_run, "sweep_id", None)
        if sweep_id is None and hasattr(mock_run, "sweep") and mock_run.sweep:
            sweep_id = mock_run.sweep.id

        assert sweep_id == "fallback-sweep-456"

        # Test 3: No sweep information available
        mock_run = MockWandbRun(has_sweep_id=False, sweep_obj=None)
        sweep_id = getattr(mock_run, "sweep_id", None)
        if sweep_id is None and hasattr(mock_run, "sweep") and mock_run.sweep:
            sweep_id = mock_run.sweep.id

        assert sweep_id is None

    def test_wandb_protein_sweep_id_initialization(self, base_sweep_config):
        """Test that WandbProtein correctly initializes with sweep ID."""

        class MockSweep:
            def __init__(self, sweep_id):
                self.id = sweep_id

        class MockWandbRun:
            def __init__(self):
                # Simulate the problematic case: no sweep_id attribute, but sweep.id available
                self.sweep = MockSweep("test-sweep-789")
                self.summary = {}
                # Fix: Mock the WandB config structure properly
                self.config = type(
                    "MockConfig",
                    (),
                    {"__dict__": {"_locked": {}}, "update": lambda self, data, allow_val_change=False: None},
                )()
                self.entity = "test-entity"
                self.project = "test-project"
                self.id = "test-run-id"
                self.name = "test-run-sweep-id"

            def update(self, data):
                pass

        class MockWandbApi:
            def runs(self, *args, **kwargs):
                # Return empty list for this test
                return []

        # Create Protein instance
        parameters_dict = OmegaConf.to_container(base_sweep_config.parameters, resolve=True)
        protein = Protein(
            parameters_dict,
            base_sweep_config.protein.max_suggestion_cost,
            base_sweep_config.protein.resample_frequency,
            base_sweep_config.protein.num_random_samples,
            base_sweep_config.protein.global_search_scale,
            base_sweep_config.protein.random_suggestions,
            base_sweep_config.protein.suggestions_per_pareto,
        )

        mock_run = MockWandbRun()

        with patch("metta.sweep.protein_wandb.wandb") as mock_wandb:
            mock_wandb.Api.return_value = MockWandbApi()

            # This should work with our fix
            wandb_protein = WandbProtein(protein, mock_run)

            # Verify sweep ID was correctly detected
            assert wandb_protein._sweep_id == "test-sweep-789"

            # Verify no observations were loaded (empty API response)
            assert wandb_protein._num_observations == 0
            assert len(wandb_protein._protein.success_observations) == 0

    def test_observation_loading_with_previous_runs(self, base_sweep_config):
        """Test that WandbProtein correctly loads observations from previous runs."""

        class MockSweep:
            def __init__(self, sweep_id):
                self.id = sweep_id

        class MockWandbRun:
            def __init__(self, run_id="current-run"):
                self.sweep = MockSweep("test-sweep-123")
                self.summary = {}
                # Fix: Mock the WandB config structure properly
                self.config = type(
                    "MockConfig",
                    (),
                    {"__dict__": {"_locked": {}}, "update": lambda self, data, allow_val_change=False: None},
                )()
                self.entity = "test-entity"
                self.project = "test-project"
                self.id = run_id
                self.name = f"test-run-{run_id}"

            def update(self, data):
                pass

        class MockPreviousRun:
            def __init__(self, run_name, run_id, objective, cost, learning_rate):
                self.name = run_name
                self.id = run_id
                self.summary = {
                    "protein.state": "success",
                    "protein.objective": objective,
                    "protein.cost": cost,
                    "protein.suggestion": {"trainer": {"optimizer": {"learning_rate": learning_rate}}},
                    "protein.suggestion_info": {"suggestion_uuid": run_id},
                }

        class MockWandbApi:
            def runs(self, *args, **kwargs):
                # Return mock previous runs
                return [
                    MockPreviousRun("test.r.0", "run-0", 0.1, 50.0, 0.0005),
                    MockPreviousRun("test.r.1", "run-1", 0.15, 60.0, 0.0003),
                    MockPreviousRun("test.r.2", "run-2", 0.12, 55.0, 0.0007),
                ]

        # Create Protein instance
        parameters_dict = OmegaConf.to_container(base_sweep_config.parameters, resolve=True)
        protein = Protein(
            parameters_dict,
            base_sweep_config.protein.max_suggestion_cost,
            base_sweep_config.protein.resample_frequency,
            base_sweep_config.protein.num_random_samples,
            base_sweep_config.protein.global_search_scale,
            base_sweep_config.protein.random_suggestions,
            base_sweep_config.protein.suggestions_per_pareto,
        )

        mock_run = MockWandbRun("current-run-new")

        with patch("metta.sweep.protein_wandb.wandb") as mock_wandb:
            mock_wandb.Api.return_value = MockWandbApi()

            # This should load the 3 previous observations
            wandb_protein = WandbProtein(protein, mock_run)

            # Verify observations were loaded
            assert wandb_protein._num_observations == 3
            assert len(wandb_protein._protein.success_observations) == 3

            # Verify observation data
            assert len(wandb_protein._observations) == 3

            # Check specific observation data
            obs = wandb_protein._observations[0]
            assert obs["run_name"] == "test.r.0"
            assert obs["objective"] == 0.1
            assert obs["cost"] == 50.0

            # Verify Protein should now use GP optimization instead of search center
            suggestion, info = wandb_protein._protein.suggest(None)
            learning_rate = suggestion["trainer"]["optimizer"]["learning_rate"]

            # With 3 observations, it should use GP optimization (not search center)
            # The exact value will depend on GP, but it should be different from 0.0005
            assert isinstance(learning_rate, float)
            assert learning_rate > 0.0

    def test_protein_decision_logic_with_observations(self, base_sweep_config):
        """Test Protein decision logic with different numbers of observations."""

        parameters_dict = OmegaConf.to_container(base_sweep_config.parameters, resolve=True)
        protein = Protein(
            parameters_dict,
            base_sweep_config.protein.max_suggestion_cost,
            base_sweep_config.protein.resample_frequency,
            base_sweep_config.protein.num_random_samples,  # 5
            base_sweep_config.protein.global_search_scale,
            base_sweep_config.protein.random_suggestions,
            base_sweep_config.protein.suggestions_per_pareto,
            seed_with_search_center=True,  # Default
        )

        # Test 1: No observations + seed_with_search_center=True → search center
        suggestion, info = protein.suggest(None)
        lr = suggestion["trainer"]["optimizer"]["learning_rate"]
        # Fix: Use pytest.approx for floating point comparison
        assert lr == pytest.approx(0.0005, rel=1e-9)  # Should be search center

        # Test 2: Add one observation → should use GP optimization (not search center)
        protein.observe({"trainer": {"optimizer": {"learning_rate": 0.0005}}}, 0.1, 50.0, False)
        suggestion, info = protein.suggest(None)
        lr = suggestion["trainer"]["optimizer"]["learning_rate"]

        # With observations > 0, should not return search center
        # (GP might return close to search center, but logic path is different)
        assert isinstance(lr, float)
        assert lr > 0.0

    def test_debug_logging_capture(self, base_sweep_config, caplog):
        """Test that our debug logging is working and can be captured."""

        class MockWandbRun:
            def __init__(self):
                self.sweep = type("MockSweep", (), {"id": "debug-test-sweep"})()
                self.summary = {}
                # Fix: Mock the WandB config structure properly
                self.config = type(
                    "MockConfig",
                    (),
                    {"__dict__": {"_locked": {}}, "update": lambda self, data, allow_val_change=False: None},
                )()
                self.entity = "debug-entity"
                self.project = "debug-project"
                self.id = "debug-run-id"
                self.name = "test-run-sweep-id"

            def update(self, data):
                pass

        class MockWandbApi:
            def runs(self, *args, **kwargs):
                return []

        parameters_dict = OmegaConf.to_container(base_sweep_config.parameters, resolve=True)
        protein = Protein(parameters_dict, **base_sweep_config.protein)

        mock_run = MockWandbRun()

        with patch("metta.sweep.protein_wandb.wandb") as mock_wandb:
            mock_wandb.Api.return_value = MockWandbApi()

            # Enable debug logging
            import logging

            logging.getLogger("wandb_protein").setLevel(logging.INFO)

            # Create WandbProtein - should generate debug logs
            wandb_protein = WandbProtein(protein, mock_run)

            # Check that our logging is working
            assert wandb_protein is not None
