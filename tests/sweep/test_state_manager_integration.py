"""Integration tests for sweep state management with existing components."""

import time
from unittest.mock import Mock, patch

import pytest
import wandb

from metta.sweep.state_manager import RobustSweepStateManager, SweepRunState


class TestStateManagerIntegration:
    """Test integration of state manager with existing sweep components."""

    @pytest.fixture
    def mock_wandb_run(self):
        """Create a realistic mock wandb run."""
        run = Mock()
        run.id = "test_run_123"
        run.name = "test_run"
        run.entity = "test_entity"
        run.project = "test_project"
        run.sweep_id = "test_sweep_456"

        # Mock summary as a dict-like object
        summary_data = {}
        run.summary = Mock()
        run.summary.update = lambda d: summary_data.update(d)
        run.summary.get = lambda k, default=None: summary_data.get(k, default)

        # Mock config - avoid setting __dict__ on Mock
        run.config = type("MockConfig", (), {"_locked": {}, "update": Mock()})()

        return run

    @pytest.fixture
    def sweep_config(self):
        """Create a basic sweep configuration."""
        return {
            "protein": {
                "num_random_samples": 3,
                "max_suggestion_cost": 300,
                "resample_frequency": 0,
                "global_search_scale": 1,
                "random_suggestions": 10,
                "suggestions_per_pareto": 5,
            },
            "parameters": {
                "metric": "reward",
                "goal": "maximize",
                "trainer": {
                    "optimizer": {
                        "learning_rate": {
                            "distribution": "log_normal",
                            "min": 0.0001,
                            "max": 0.01,
                            "mean": 0.001,
                            "scale": 0.5,
                        }
                    }
                },
            },
        }

    def test_state_manager_with_protein_wandb_lifecycle(self, mock_wandb_run, sweep_config):
        """Test state manager integration with protein_wandb lifecycle."""
        # Create state manager
        state_manager = RobustSweepStateManager(mock_wandb_run, timeout_minutes=30)

        # Simulate protein_wandb initialization
        state_manager.set_state(SweepRunState.INITIALIZING)

        # Mock protein creation
        with patch("metta.sweep.protein_metta.Protein") as mock_protein_class:
            mock_protein = Mock()
            mock_protein.suggest.return_value = ({"trainer/optimizer/learning_rate": 0.001}, {"cost": 100.0})
            mock_protein.observe = Mock()
            mock_protein_class.return_value = mock_protein

            with patch("wandb.Api") as mock_api:
                mock_api.return_value.runs.return_value = []

                # This would normally be done inside protein_wandb
                try:
                    # Load previous runs
                    state_manager.set_state(SweepRunState.RUNNING)

                    # Simulate training
                    for _ in range(3):
                        time.sleep(0.05)
                        state_manager.heartbeat()

                    # Simulate evaluation
                    state_manager.set_state(SweepRunState.EVALUATING)

                    # Record observation
                    objective = 0.85
                    cost = 120.0

                    # Success
                    state_manager.set_state(SweepRunState.SUCCESS, objective=objective, cost=cost)

                except Exception as e:
                    state_manager.handle_exception(e)

        # Verify state progression
        assert mock_wandb_run.summary.get("protein.state") == "success"
        assert mock_wandb_run.summary.get("protein.objective") == 0.85
        assert mock_wandb_run.summary.get("protein.cost") == 120.0

    def test_state_manager_with_training_failure(self, mock_wandb_run):
        """Test state manager handling training failures."""
        state_manager = RobustSweepStateManager(mock_wandb_run)

        # Use context manager for automatic state handling
        with pytest.raises(RuntimeError):
            with state_manager:
                # Simulate some training progress
                time.sleep(0.05)
                state_manager.heartbeat()

                # Training fails
                raise RuntimeError("CUDA out of memory")

        # Verify failure state
        assert mock_wandb_run.summary.get("protein.state") == "failure"
        assert "RuntimeError: CUDA out of memory" in mock_wandb_run.summary.get("protein.error", "")

    def test_state_manager_with_timeout_detection(self, mock_wandb_run):
        """Test timeout detection during long-running operations."""
        # Create manager with very short timeout for testing
        state_manager = RobustSweepStateManager(mock_wandb_run, timeout_minutes=0.01)  # 0.6 seconds

        state_manager.set_state(SweepRunState.RUNNING)

        # Simulate long operation without heartbeat
        time.sleep(0.7)

        # Check timeout
        assert state_manager.check_timeout()

        # Handle timeout
        state_manager.set_state(SweepRunState.TIMEOUT, error="Training exceeded timeout without heartbeat")

        assert mock_wandb_run.summary.get("protein.state") == "timeout"

    def test_state_manager_in_sweep_eval_workflow(self, mock_wandb_run):
        """Test state manager in sweep evaluation workflow."""
        state_manager = RobustSweepStateManager(mock_wandb_run)

        # Simulate sweep_eval.py workflow
        try:
            # Start evaluation
            state_manager.set_state(SweepRunState.EVALUATING)

            # Mock policy loading
            time.sleep(0.05)

            # Mock evaluation
            eval_metric = 0.92
            eval_time = 15.0

            # Record success
            state_manager.set_state(SweepRunState.SUCCESS, score=eval_metric, eval_time=eval_time)

        except Exception as e:
            state_manager.handle_exception(e)

        assert mock_wandb_run.summary.get("protein.state") == "success"
        assert mock_wandb_run.summary.get("protein.score") == 0.92

    def test_state_manager_prevents_stuck_runs(self, mock_wandb_run):
        """Test that state manager helps prevent stuck runs."""
        state_manager = RobustSweepStateManager(mock_wandb_run, timeout_minutes=30)

        # Track heartbeats
        heartbeat_times = []

        # Simulate training with periodic heartbeats
        state_manager.set_state(SweepRunState.RUNNING)

        for _ in range(5):
            time.sleep(0.1)
            state_manager.heartbeat()
            heartbeat_times.append(mock_wandb_run.summary.get("protein.heartbeat"))

        # All heartbeats should be unique timestamps
        assert len(set(heartbeat_times)) == 5

        # Should not be timed out
        assert not state_manager.check_timeout()

    def test_state_manager_with_wandb_offline_mode(self):
        """Test state manager works in wandb offline mode."""
        import os

        os.environ["WANDB_MODE"] = "offline"

        wandb.init(project="test_project", mode="offline")

        try:
            state_manager = RobustSweepStateManager(wandb.run)

            # Should work even in offline mode
            state_manager.set_state(SweepRunState.INITIALIZING)
            state_manager.set_state(SweepRunState.RUNNING)
            state_manager.heartbeat()
            state_manager.set_state(SweepRunState.SUCCESS, score=0.88)

            # Verify states were set
            assert wandb.run.summary.get("protein.state") == "success"
            assert wandb.run.summary.get("protein.score") == 0.88

        finally:
            wandb.finish()

    def test_state_manager_error_recovery(self, mock_wandb_run):
        """Test state manager continues working after wandb errors."""
        # Make wandb operations flaky
        call_count = 0

        def flaky_update(data):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Fail every other call
                raise RuntimeError("WandB API error")
            mock_wandb_run.summary.__dict__.update(data)

        mock_wandb_run.summary.update = flaky_update

        state_manager = RobustSweepStateManager(mock_wandb_run)

        # These should not raise despite flaky wandb
        state_manager.set_state(SweepRunState.INITIALIZING)  # Succeeds
        state_manager.set_state(SweepRunState.RUNNING)  # Fails but handled
        state_manager.heartbeat()  # Succeeds
        state_manager.heartbeat()  # Fails but handled

        # Should have attempted all operations
        assert call_count == 4

    def test_state_transitions_are_logged(self, mock_wandb_run, caplog):
        """Test that state transitions are properly logged."""
        import logging

        caplog.set_level(logging.INFO)

        state_manager = RobustSweepStateManager(mock_wandb_run)

        # Make state transitions
        state_manager.set_state(SweepRunState.INITIALIZING)
        state_manager.set_state(SweepRunState.RUNNING)
        state_manager.set_state(SweepRunState.SUCCESS, score=0.95)

        # Check logs
        log_messages = [record.message for record in caplog.records]
        assert any("initializing" in msg for msg in log_messages)
        assert any("running" in msg for msg in log_messages)
        assert any("success" in msg for msg in log_messages)
