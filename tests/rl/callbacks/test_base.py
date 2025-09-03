"""Unit tests for TrainerCallback base class."""

import pytest
import torch

from metta.rl.callbacks.base import TrainerCallback
from metta.rl.trainer_state import TrainerState


class TestTrainerCallback:
    """Test suite for TrainerCallback base class."""

    def test_callback_initialization(self):
        """Test that a callback can be initialized with required parameters."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        
        callback = TrainerCallback(
            trainer_cfg=trainer_cfg,
            device=device,
            instance_name="test_callback",
            critical=False
        )
        
        assert callback.trainer_cfg is trainer_cfg
        assert callback.device == device
        assert callback.instance_name == "test_callback"
        assert callback.critical is False

    def test_callback_initialization_defaults(self):
        """Test callback initialization with default values."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        
        callback = TrainerCallback(trainer_cfg=trainer_cfg, device=device)
        
        assert callback.instance_name == "callback"
        assert callback.critical is False

    def test_all_callbacks_exist(self):
        """Test that all expected callbacks are defined."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        callback = TrainerCallback(trainer_cfg=trainer_cfg, device=device)
        trainer_state = TrainerState()
        
        # All callbacks should be callable and not raise
        callback.on_new_training_run(trainer_state)
        callback.on_rollout_start(trainer_state)
        callback.on_rollout_end(trainer_state)
        callback.on_mb_end(trainer_state)
        callback.on_train_phase_end(trainer_state)
        callback.on_epoch_end(trainer_state)
        callback.on_training_end(trainer_state)

    def test_should_run_utility(self):
        """Test the should_run utility method."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        callback = TrainerCallback(trainer_cfg=trainer_cfg, device=device)
        
        # Test with interval 0 (disabled)
        assert callback.should_run(epoch=10, interval=0) is False
        
        # Test with interval 1 (every epoch)
        assert callback.should_run(epoch=0, interval=1) is False  # Epoch 0 doesn't run
        assert callback.should_run(epoch=1, interval=1) is True
        assert callback.should_run(epoch=2, interval=1) is True
        
        # Test with interval 5 (every 5 epochs)
        assert callback.should_run(epoch=0, interval=5) is False
        assert callback.should_run(epoch=4, interval=5) is False
        assert callback.should_run(epoch=5, interval=5) is True
        assert callback.should_run(epoch=10, interval=5) is True
        assert callback.should_run(epoch=11, interval=5) is False

    def test_callback_repr(self):
        """Test the string representation of a callback."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        callback = TrainerCallback(
            trainer_cfg=trainer_cfg,
            device=device,
            instance_name="my_test_callback"
        )
        
        assert repr(callback) == "TrainerCallback(instance_name='my_test_callback')"


class MockCallback(TrainerCallback):
    """Mock callback for testing inheritance and callback tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
    
    def on_new_training_run(self, trainer_state: TrainerState) -> None:
        self.calls.append('on_new_training_run')
    
    def on_rollout_start(self, trainer_state: TrainerState) -> None:
        self.calls.append('on_rollout_start')
    
    def on_rollout_end(self, trainer_state: TrainerState) -> None:
        self.calls.append('on_rollout_end')
        # Access trainer state data
        if trainer_state.rollout_stats:
            self.calls.append('has_rollout_stats')
    
    def on_mb_end(self, trainer_state: TrainerState) -> None:
        self.calls.append('on_mb_end')
    
    def on_train_phase_end(self, trainer_state: TrainerState) -> None:
        self.calls.append('on_train_phase_end')
    
    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        self.calls.append('on_epoch_end')
        # Access trainer state data
        if trainer_state.loss_stats:
            self.calls.append('has_loss_stats')
        if trainer_state.eval_scores:
            self.calls.append('has_eval_scores')
    
    def on_training_end(self, trainer_state: TrainerState) -> None:
        self.calls.append('on_training_end')


class TestMockCallback:
    """Test callback inheritance and callback behavior."""
    
    def test_mock_callback_callbacks(self):
        """Test that a custom callback can override callbacks."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        callback = MockCallback(trainer_cfg=trainer_cfg, device=device)
        
        trainer_state = TrainerState()
        
        # Test callbacks are tracked
        callback.on_new_training_run(trainer_state)
        assert 'on_new_training_run' in callback.calls
        
        callback.on_rollout_start(trainer_state)
        assert 'on_rollout_start' in callback.calls
        
        # Test accessing trainer state data
        trainer_state.rollout_stats = {'test': [1, 2, 3]}
        callback.on_rollout_end(trainer_state)
        assert 'on_rollout_end' in callback.calls
        assert 'has_rollout_stats' in callback.calls
        
        # Test epoch end with loss stats
        trainer_state.loss_stats = {'policy_loss': 0.5}
        trainer_state.eval_scores = type('EvalScores', (), {'mean_reward': 10.0})()
        callback.on_epoch_end(trainer_state)
        assert 'on_epoch_end' in callback.calls
        assert 'has_loss_stats' in callback.calls
        assert 'has_eval_scores' in callback.calls

    def test_critical_callback_behavior(self):
        """Test that critical callbacks behave differently."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        
        # Non-critical callback
        non_critical_callback = MockCallback(
            trainer_cfg=trainer_cfg,
            device=device,
            critical=False
        )
        assert non_critical_callback.critical is False
        
        # Critical callback
        critical_callback = MockCallback(
            trainer_cfg=trainer_cfg,
            device=device,
            critical=True
        )
        assert critical_callback.critical is True