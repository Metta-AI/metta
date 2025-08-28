"""Tests for auto_config functions."""

from unittest.mock import Mock, patch

from metta.tools.utils.auto_config import auto_replay_dir, auto_torch_profile_dir


class TestAutoConfig:
    """Test auto configuration functions."""

    def test_auto_replay_dir_external_user(self):
        """External users should get local replay directory."""
        with patch("metta.tools.utils.auto_config.AWSSetup") as mock_aws_setup:
            mock_setup = Mock()
            mock_setup.to_config_settings.return_value = {"replay_dir": "./train_dir/replays/"}
            mock_aws_setup.return_value = mock_setup

            result = auto_replay_dir()
            assert result == "./train_dir/replays/"


    def test_auto_torch_profile_dir_external_user(self):
        """External users should get local torch profile directory."""
        with patch("metta.setup.saved_settings.get_saved_settings") as mock_settings:
            mock_saved = Mock()
            mock_saved.user_type.is_softmax = False
            mock_settings.return_value = mock_saved

            result = auto_torch_profile_dir()
            assert result == "./train_dir/torch_traces/"

    def test_auto_torch_profile_dir_softmax_user(self):
        """Softmax users should get S3 torch profile directory."""
        with patch("metta.setup.saved_settings.get_saved_settings") as mock_settings:
            mock_saved = Mock()
            mock_saved.user_type.is_softmax = True
            mock_settings.return_value = mock_saved

            result = auto_torch_profile_dir()
            assert result == "s3://softmax-public/torch_traces/"

