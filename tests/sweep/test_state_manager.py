"""Unit tests for sweep state management."""

import time
from datetime import datetime
from unittest.mock import Mock

import pytest

from metta.sweep.state_manager import RobustSweepStateManager, SweepRunState


class TestSweepRunState:
    """Test the SweepRunState enum."""

    def test_state_values(self):
        """Test that enum values are correctly defined."""
        assert SweepRunState.INITIALIZING.value == "initializing"
        assert SweepRunState.RUNNING.value == "running"
        assert SweepRunState.EVALUATING.value == "evaluating"
        assert SweepRunState.SUCCESS.value == "success"
        assert SweepRunState.FAILURE.value == "failure"
        assert SweepRunState.TIMEOUT.value == "timeout"
        assert SweepRunState.CRASHED.value == "crashed"
        assert SweepRunState.CANCELLED.value == "cancelled"

    def test_all_states_defined(self):
        """Test that all expected states are defined."""
        expected_states = {
            "INITIALIZING",
            "RUNNING",
            "EVALUATING",
            "SUCCESS",
            "FAILURE",
            "TIMEOUT",
            "CRASHED",
            "CANCELLED",
        }
        actual_states = {state.name for state in SweepRunState}
        assert actual_states == expected_states


class TestRobustSweepStateManager:
    """Test the RobustSweepStateManager class."""

    @pytest.fixture
    def mock_wandb_run(self):
        """Create a mock wandb run."""
        run = Mock()
        run.summary = Mock()
        run.summary.update = Mock()
        run.summary.get = Mock(return_value=None)
        return run

    @pytest.fixture
    def state_manager(self, mock_wandb_run):
        """Create a state manager with mock wandb run."""
        return RobustSweepStateManager(mock_wandb_run, timeout_minutes=30)

    def test_initialization(self, mock_wandb_run):
        """Test state manager initialization."""
        manager = RobustSweepStateManager(mock_wandb_run, timeout_minutes=45)

        assert manager.wandb_run == mock_wandb_run
        assert manager.timeout_seconds == 45 * 60
        assert isinstance(manager.start_time, float)
        assert isinstance(manager._last_heartbeat, float)

    def test_set_state_basic(self, state_manager, mock_wandb_run):
        """Test basic state setting."""
        state_manager.set_state(SweepRunState.RUNNING)

        # Verify update was called
        mock_wandb_run.summary.update.assert_called_once()
        update_dict = mock_wandb_run.summary.update.call_args[0][0]

        # Check required fields
        assert update_dict["protein.state"] == "running"
        assert "protein.last_update" in update_dict
        assert "protein.runtime_seconds" in update_dict

        # Verify timestamp format
        timestamp = update_dict["protein.last_update"]
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    def test_set_state_with_error(self, state_manager, mock_wandb_run):
        """Test state setting with error message."""
        state_manager.set_state(SweepRunState.FAILURE, error="Test error message")

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert update_dict["protein.state"] == "failure"
        assert update_dict["protein.error"] == "Test error message"

    def test_set_state_with_extra_fields(self, state_manager, mock_wandb_run):
        """Test state setting with extra fields."""
        state_manager.set_state(
            SweepRunState.SUCCESS, score=0.95, train_time=120.5, **{"protein.custom_field": "value"}
        )

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert update_dict["protein.state"] == "success"
        assert update_dict["protein.score"] == 0.95
        assert update_dict["protein.train_time"] == 120.5
        assert update_dict["protein.custom_field"] == "value"

    def test_heartbeat(self, state_manager, mock_wandb_run):
        """Test heartbeat functionality."""
        # Sleep briefly to ensure time difference
        time.sleep(0.1)

        state_manager.heartbeat()

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert "protein.heartbeat" in update_dict
        assert "protein.runtime_seconds" in update_dict

        # Verify heartbeat timestamp
        timestamp = update_dict["protein.heartbeat"]
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

        # Verify internal heartbeat tracking
        assert state_manager._last_heartbeat > state_manager.start_time

    def test_check_timeout(self, state_manager):
        """Test timeout checking."""
        # Initially should not be timed out
        assert not state_manager.check_timeout()

        # Manually set last heartbeat to past
        state_manager._last_heartbeat = time.time() - (31 * 60)  # 31 minutes ago
        assert state_manager.check_timeout()

        # Reset heartbeat
        state_manager.heartbeat()
        assert not state_manager.check_timeout()

    def test_get_runtime_seconds(self, state_manager):
        """Test runtime calculation."""
        time.sleep(0.1)  # Sleep briefly

        runtime = state_manager.get_runtime_seconds()
        assert runtime > 0
        assert runtime < 1  # Should be less than 1 second for test

    def test_transition_to_terminal_state_success(self, state_manager, mock_wandb_run):
        """Test transition to success state."""
        state_manager.transition_to_terminal_state(success=True, score=0.88, total_time=150.0)

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert update_dict["protein.state"] == "success"
        assert update_dict["protein.score"] == 0.88
        assert update_dict["protein.total_time"] == 150.0
        assert "protein.error" not in update_dict

    def test_transition_to_terminal_state_failure(self, state_manager, mock_wandb_run):
        """Test transition to failure state."""
        state_manager.transition_to_terminal_state(success=False, error="Training failed", last_epoch=10)

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert update_dict["protein.state"] == "failure"
        assert update_dict["protein.error"] == "Training failed"
        assert update_dict["protein.last_epoch"] == 10

    def test_handle_exception_timeout(self, state_manager, mock_wandb_run):
        """Test handling timeout exception."""
        exc = TimeoutError("Operation timed out")
        state_manager.handle_exception(exc)

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert update_dict["protein.state"] == "timeout"
        assert "TimeoutError: Operation timed out" in update_dict["protein.error"]

    def test_handle_exception_keyboard_interrupt(self, state_manager, mock_wandb_run):
        """Test handling keyboard interrupt."""
        exc = KeyboardInterrupt("User cancelled")
        state_manager.handle_exception(exc)

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert update_dict["protein.state"] == "cancelled"
        assert "KeyboardInterrupt" in update_dict["protein.error"]

    def test_handle_exception_generic(self, state_manager, mock_wandb_run):
        """Test handling generic exception."""
        exc = ValueError("Invalid value")
        state_manager.handle_exception(exc)

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert update_dict["protein.state"] == "failure"
        assert "ValueError: Invalid value" in update_dict["protein.error"]

    def test_handle_exception_with_custom_state(self, state_manager, mock_wandb_run):
        """Test handling exception with custom state."""
        exc = RuntimeError("System crashed")
        state_manager.handle_exception(exc, state=SweepRunState.CRASHED)

        update_dict = mock_wandb_run.summary.update.call_args[0][0]
        assert update_dict["protein.state"] == "crashed"
        assert "RuntimeError: System crashed" in update_dict["protein.error"]

    def test_context_manager_success(self, state_manager, mock_wandb_run):
        """Test context manager with successful execution."""
        with state_manager:
            # Should set RUNNING state on entry
            first_call = mock_wandb_run.summary.update.call_args_list[0][0][0]
            assert first_call["protein.state"] == "running"

            # Do some work
            time.sleep(0.05)

        # Should set SUCCESS state on exit
        last_call = mock_wandb_run.summary.update.call_args_list[-1][0][0]
        assert last_call["protein.state"] == "success"

    def test_context_manager_with_exception(self, state_manager, mock_wandb_run):
        """Test context manager with exception."""
        with pytest.raises(ValueError):
            with state_manager:
                # Should set RUNNING state
                first_call = mock_wandb_run.summary.update.call_args_list[0][0][0]
                assert first_call["protein.state"] == "running"

                # Raise exception
                raise ValueError("Test error")

        # Should set FAILURE state
        last_call = mock_wandb_run.summary.update.call_args_list[-1][0][0]
        assert last_call["protein.state"] == "failure"
        assert "ValueError: Test error" in last_call["protein.error"]

    def test_context_manager_no_double_success(self, state_manager, mock_wandb_run):
        """Test that context manager doesn't override terminal states."""
        # Set run to already be in success state
        mock_wandb_run.summary.get.return_value = "success"

        with state_manager:
            pass

        # Should have set RUNNING on entry, but not SUCCESS on exit
        calls = mock_wandb_run.summary.update.call_args_list
        states = [call[0][0]["protein.state"] for call in calls]
        assert states == ["running"]  # Only RUNNING, no SUCCESS

    def test_error_handling_in_set_state(self, mock_wandb_run):
        """Test error handling when wandb update fails."""
        # Make update raise exception
        mock_wandb_run.summary.update.side_effect = RuntimeError("WandB error")

        manager = RobustSweepStateManager(mock_wandb_run)

        # Should not raise, just log error
        manager.set_state(SweepRunState.RUNNING)  # Should not raise

    def test_error_handling_in_heartbeat(self, mock_wandb_run):
        """Test error handling when heartbeat fails."""
        mock_wandb_run.summary.update.side_effect = RuntimeError("WandB error")

        manager = RobustSweepStateManager(mock_wandb_run)

        # Should not raise, just log error
        manager.heartbeat()  # Should not raise

    def test_state_manager_with_real_time_progression(self, state_manager, mock_wandb_run):
        """Test state manager with realistic time progression."""
        # Initialize
        state_manager.set_state(SweepRunState.INITIALIZING)

        # Start running
        time.sleep(0.1)
        state_manager.set_state(SweepRunState.RUNNING)

        # Send heartbeats during training
        for _ in range(3):
            time.sleep(0.05)
            state_manager.heartbeat()

        # Start evaluation
        state_manager.set_state(SweepRunState.EVALUATING)
        time.sleep(0.05)

        # Complete successfully
        state_manager.set_state(SweepRunState.SUCCESS, score=0.92, total_time=state_manager.get_runtime_seconds())

        # Verify state progression
        calls = mock_wandb_run.summary.update.call_args_list
        states = [call[0][0].get("protein.state") for call in calls if "protein.state" in call[0][0]]

        assert states == ["initializing", "running", "evaluating", "success"]

        # Verify heartbeats were sent
        heartbeat_calls = [call for call in calls if "protein.heartbeat" in call[0][0]]
        assert len(heartbeat_calls) == 3
