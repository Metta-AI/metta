"""Unit tests for TrainerHook base class."""

import pytest
import torch

from metta.rl.hooks.base import TrainerHook
from metta.rl.trainer_state import TrainerState


class TestTrainerHook:
    """Test suite for TrainerHook base class."""

    def test_hook_initialization(self):
        """Test that a hook can be initialized with required parameters."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        
        hook = TrainerHook(
            trainer_cfg=trainer_cfg,
            device=device,
            instance_name="test_hook",
            critical=False
        )
        
        assert hook.trainer_cfg is trainer_cfg
        assert hook.device == device
        assert hook.instance_name == "test_hook"
        assert hook.critical is False

    def test_hook_initialization_defaults(self):
        """Test hook initialization with default values."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        
        hook = TrainerHook(trainer_cfg=trainer_cfg, device=device)
        
        assert hook.instance_name == "hook"
        assert hook.critical is False

    def test_all_callbacks_exist(self):
        """Test that all expected callbacks are defined."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        hook = TrainerHook(trainer_cfg=trainer_cfg, device=device)
        trainer_state = TrainerState()
        
        # All callbacks should be callable and not raise
        hook.on_new_training_run(trainer_state)
        hook.on_rollout_start(trainer_state)
        hook.on_rollout_end(trainer_state)
        hook.on_mb_end(trainer_state)
        hook.on_train_phase_end(trainer_state)
        hook.on_epoch_end(trainer_state)
        hook.on_training_end(trainer_state)

    def test_should_run_utility(self):
        """Test the should_run utility method."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        hook = TrainerHook(trainer_cfg=trainer_cfg, device=device)
        
        # Test with interval 0 (disabled)
        assert hook.should_run(epoch=10, interval=0) is False
        
        # Test with interval 1 (every epoch)
        assert hook.should_run(epoch=0, interval=1) is False  # Epoch 0 doesn't run
        assert hook.should_run(epoch=1, interval=1) is True
        assert hook.should_run(epoch=2, interval=1) is True
        
        # Test with interval 5 (every 5 epochs)
        assert hook.should_run(epoch=0, interval=5) is False
        assert hook.should_run(epoch=4, interval=5) is False
        assert hook.should_run(epoch=5, interval=5) is True
        assert hook.should_run(epoch=10, interval=5) is True
        assert hook.should_run(epoch=11, interval=5) is False

    def test_hook_repr(self):
        """Test the string representation of a hook."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        hook = TrainerHook(
            trainer_cfg=trainer_cfg,
            device=device,
            instance_name="my_test_hook"
        )
        
        assert repr(hook) == "TrainerHook(instance_name='my_test_hook')"


class MockHook(TrainerHook):
    """Mock hook for testing inheritance and callback tracking."""
    
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


class TestMockHook:
    """Test hook inheritance and callback behavior."""
    
    def test_mock_hook_callbacks(self):
        """Test that a custom hook can override callbacks."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        hook = MockHook(trainer_cfg=trainer_cfg, device=device)
        
        trainer_state = TrainerState()
        
        # Test callbacks are tracked
        hook.on_new_training_run(trainer_state)
        assert 'on_new_training_run' in hook.calls
        
        hook.on_rollout_start(trainer_state)
        assert 'on_rollout_start' in hook.calls
        
        # Test accessing trainer state data
        trainer_state.rollout_stats = {'test': [1, 2, 3]}
        hook.on_rollout_end(trainer_state)
        assert 'on_rollout_end' in hook.calls
        assert 'has_rollout_stats' in hook.calls
        
        # Test epoch end with loss stats
        trainer_state.loss_stats = {'policy_loss': 0.5}
        trainer_state.eval_scores = type('EvalScores', (), {'mean_reward': 10.0})()
        hook.on_epoch_end(trainer_state)
        assert 'on_epoch_end' in hook.calls
        assert 'has_loss_stats' in hook.calls
        assert 'has_eval_scores' in hook.calls

    def test_critical_hook_behavior(self):
        """Test that critical hooks behave differently."""
        trainer_cfg = type('TrainerConfig', (), {})()
        device = torch.device('cpu')
        
        # Non-critical hook
        non_critical_hook = MockHook(
            trainer_cfg=trainer_cfg,
            device=device,
            critical=False
        )
        assert non_critical_hook.critical is False
        
        # Critical hook
        critical_hook = MockHook(
            trainer_cfg=trainer_cfg,
            device=device,
            critical=True
        )
        assert critical_hook.critical is True