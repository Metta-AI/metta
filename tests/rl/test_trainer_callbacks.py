"""Test that new trainer callbacks are invoked correctly."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from metta.rl.loss.base_loss import BaseLoss
from metta.rl.trainer_state import TrainerState


class CallbackTrackingLoss(BaseLoss):
    """Test loss that tracks callback invocations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_counts = {
            'on_new_training_run': 0,
            'on_rollout_start': 0,
            'on_rollout_end': 0,
            'on_train_phase_end': 0,
            'on_epoch_end': 0,
            'on_mb_end': 0,
        }
        self.last_trainer_state = None
    
    def on_new_training_run(self, trainer_state: TrainerState) -> None:
        super().on_new_training_run(trainer_state)
        self.callback_counts['on_new_training_run'] += 1
        self.last_trainer_state = trainer_state
    
    def on_rollout_start(self, trainer_state: TrainerState) -> None:
        super().on_rollout_start(trainer_state)
        self.callback_counts['on_rollout_start'] += 1
        self.last_trainer_state = trainer_state
    
    def on_rollout_end(self, trainer_state: TrainerState) -> None:
        super().on_rollout_end(trainer_state)
        self.callback_counts['on_rollout_end'] += 1
        self.last_trainer_state = trainer_state
        # Verify rollout_stats is populated
        assert trainer_state.rollout_stats is not None, "rollout_stats should be populated"
    
    def on_train_phase_end(self, trainer_state: TrainerState) -> None:
        super().on_train_phase_end(trainer_state)
        self.callback_counts['on_train_phase_end'] += 1
        self.last_trainer_state = trainer_state
    
    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        super().on_epoch_end(trainer_state)
        self.callback_counts['on_epoch_end'] += 1
        self.last_trainer_state = trainer_state
        # Verify loss_stats is populated
        assert trainer_state.loss_stats is not None, "loss_stats should be populated"
    
    def on_mb_end(self, trainer_state: TrainerState) -> None:
        super().on_mb_end(trainer_state)
        self.callback_counts['on_mb_end'] += 1
        self.last_trainer_state = trainer_state


class TestTrainerCallbacks(unittest.TestCase):
    """Test that trainer callbacks are invoked correctly."""
    
    def test_trainer_state_has_new_fields(self):
        """Test that TrainerState has the new hook-related fields."""
        trainer_state = TrainerState()
        
        # Check that all new fields exist
        self.assertTrue(hasattr(trainer_state, 'rollout_stats'))
        self.assertTrue(hasattr(trainer_state, 'loss_stats'))
        self.assertTrue(hasattr(trainer_state, 'eval_scores'))
        self.assertTrue(hasattr(trainer_state, 'experience'))
        self.assertTrue(hasattr(trainer_state, 'policy'))
        self.assertTrue(hasattr(trainer_state, 'latest_checkpoint_uri'))
        self.assertTrue(hasattr(trainer_state, 'latest_wandb_uri'))
        self.assertTrue(hasattr(trainer_state, 'stats_tracker'))
        self.assertTrue(hasattr(trainer_state, 'timer'))
        
        # Check initial values are None
        self.assertIsNone(trainer_state.rollout_stats)
        self.assertIsNone(trainer_state.loss_stats)
        self.assertIsNone(trainer_state.eval_scores)
        self.assertIsNone(trainer_state.experience)
        self.assertIsNone(trainer_state.policy)
        self.assertIsNone(trainer_state.latest_checkpoint_uri)
        self.assertIsNone(trainer_state.latest_wandb_uri)
        self.assertIsNone(trainer_state.stats_tracker)
        self.assertIsNone(trainer_state.timer)
    
    def test_base_loss_has_new_callbacks(self):
        """Test that BaseLoss has the new callback methods."""
        # Create a mock loss instance
        mock_policy = MagicMock()
        mock_trainer_cfg = MagicMock()
        mock_vec_env = MagicMock()
        mock_device = torch.device('cpu')
        mock_checkpoint_manager = MagicMock()
        mock_loss_config = MagicMock()
        
        loss = BaseLoss(
            policy=mock_policy,
            trainer_cfg=mock_trainer_cfg,
            vec_env=mock_vec_env,
            device=mock_device,
            checkpoint_manager=mock_checkpoint_manager,
            instance_name='test_loss',
            loss_config=mock_loss_config
        )
        
        # Check that new methods exist
        self.assertTrue(hasattr(loss, 'on_rollout_end'))
        self.assertTrue(hasattr(loss, 'on_epoch_end'))
        
        # Test that methods can be called without error
        trainer_state = TrainerState()
        loss.on_rollout_end(trainer_state)  # Should not raise
        loss.on_epoch_end(trainer_state)  # Should not raise
    
    def test_callback_invocation_order(self):
        """Test that callbacks would be invoked in the correct order."""
        # This is a conceptual test - in real training, the order would be:
        # 1. on_new_training_run (once at start)
        # 2. on_rollout_start (before each rollout)
        # 3. on_rollout_end (after rollout, before training)
        # 4. on_mb_end (after each minibatch)
        # 5. on_train_phase_end (after training phase)
        # 6. on_epoch_end (after epoch)
        
        mock_policy = MagicMock()
        mock_trainer_cfg = MagicMock()
        mock_vec_env = MagicMock()
        mock_device = torch.device('cpu')
        mock_checkpoint_manager = MagicMock()
        mock_loss_config = MagicMock()
        
        tracking_loss = CallbackTrackingLoss(
            policy=mock_policy,
            trainer_cfg=mock_trainer_cfg,
            vec_env=mock_vec_env,
            device=mock_device,
            checkpoint_manager=mock_checkpoint_manager,
            instance_name='tracking_loss',
            loss_config=mock_loss_config
        )
        
        trainer_state = TrainerState(
            rollout_stats={'test': [1.0]},
            loss_stats={'loss': 0.5}
        )
        
        # Simulate a training sequence
        tracking_loss.on_new_training_run(trainer_state)
        tracking_loss.on_rollout_start(trainer_state)
        tracking_loss.on_rollout_end(trainer_state)
        tracking_loss.on_mb_end(trainer_state)
        tracking_loss.on_train_phase_end(trainer_state)
        tracking_loss.on_epoch_end(trainer_state)
        
        # Verify all callbacks were called
        self.assertEqual(tracking_loss.callback_counts['on_new_training_run'], 1)
        self.assertEqual(tracking_loss.callback_counts['on_rollout_start'], 1)
        self.assertEqual(tracking_loss.callback_counts['on_rollout_end'], 1)
        self.assertEqual(tracking_loss.callback_counts['on_mb_end'], 1)
        self.assertEqual(tracking_loss.callback_counts['on_train_phase_end'], 1)
        self.assertEqual(tracking_loss.callback_counts['on_epoch_end'], 1)


if __name__ == '__main__':
    unittest.main()