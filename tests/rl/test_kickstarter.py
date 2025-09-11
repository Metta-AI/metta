from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

from metta.rl.kickstarter import Kickstarter
from metta.rl.kickstarter_config import KickstartTeacherConfig


class TestKickstarter:
    @pytest.fixture
    def mock_config(self):
        cfg = MagicMock()
        cfg.additional_teachers = None
        cfg.anneal_ratio = 0.2
        cfg.teacher_uri = None
        cfg.kickstart_steps = 1000
        cfg.action_loss_coef = 0.5
        cfg.value_loss_coef = 0.5

        return cfg

    @pytest.fixture
    def mock_policy_store(self):
        return MagicMock()

    @pytest.fixture
    def mock_metta_grid_env(self):
        env = MagicMock()
        env.action_names = ["move", "attack"]
        env.max_action_args = [4, 2]
        env.get_observation_features.return_value = MagicMock()  # Mock features
        return env

    def test_initialization_no_teachers(self, mock_config, mock_policy_store, mock_metta_grid_env):
        kickstarter = Kickstarter(mock_config, "cpu", mock_policy_store, mock_metta_grid_env)

        assert kickstarter.enabled is False
        assert kickstarter.anneal_ratio == 0.2
        assert kickstarter.device == "cpu"

    @patch("metta.rl.kickstarter.Kickstarter._load_policies")
    def test_initialization_with_teacher_uri(
        self, mock_load_policies, mock_config, mock_policy_store, mock_metta_grid_env
    ):
        """Test initialization when a teacher URI is provided."""
        mock_config.teacher_uri = "wandb://teacher/uri"

        kickstarter = Kickstarter(mock_config, "cpu", mock_policy_store, mock_metta_grid_env)

        assert kickstarter.enabled is True
        assert kickstarter.teacher_cfgs is not None
        assert len(kickstarter.teacher_cfgs) == 1
        assert kickstarter.teacher_cfgs[0].teacher_uri == "wandb://teacher/uri"
        assert kickstarter.teacher_cfgs[0].action_loss_coef == 0.5
        assert kickstarter.teacher_cfgs[0].value_loss_coef == 0.5

    @patch("metta.rl.kickstarter.Kickstarter._load_policies")
    def test_initialization_with_additional_teachers(
        self, mock_load_policies, mock_config, mock_policy_store, mock_metta_grid_env
    ):
        """Test initialization when additional teachers are provided."""
        mock_config.additional_teachers = [
            KickstartTeacherConfig(teacher_uri="wandb://teacher1/uri", action_loss_coef=0.3, value_loss_coef=0.7),
            KickstartTeacherConfig(teacher_uri="wandb://teacher2/uri", action_loss_coef=0.6, value_loss_coef=0.4),
        ]

        kickstarter = Kickstarter(mock_config, "cpu", mock_policy_store, mock_metta_grid_env)

        assert kickstarter.enabled is True
        assert kickstarter.teacher_cfgs is not None
        assert len(kickstarter.teacher_cfgs) == 2
        assert kickstarter.teacher_cfgs[0].teacher_uri == "wandb://teacher1/uri"
        assert kickstarter.teacher_cfgs[1].teacher_uri == "wandb://teacher2/uri"

    @patch("metta.rl.kickstarter.Kickstarter._load_policies")
    def test_anneal_factor_calculation(self, mock_load_policies, mock_config, mock_policy_store, mock_metta_grid_env):
        """Test the calculation of the anneal factor."""
        mock_config.teacher_uri = "wandb://teacher/uri"

        kickstarter = Kickstarter(mock_config, "cpu", mock_policy_store, mock_metta_grid_env)

        # Initial anneal factor should be 1.0
        assert kickstarter.anneal_factor == 1.0

        # Calculate expected values
        kickstart_steps = 1000
        anneal_ratio = 0.2
        anneal_duration = kickstart_steps * anneal_ratio  # 200
        ramp_down_start_step = kickstart_steps - anneal_duration  # 800

        assert kickstarter.kickstart_steps == kickstart_steps
        assert kickstarter.anneal_duration == anneal_duration
        assert kickstarter.ramp_down_start_step == ramp_down_start_step

    @patch("metta.rl.kickstarter.Kickstarter._load_policies")
    def test_loss_disabled(self, mock_load_policies, mock_config, mock_policy_store, mock_metta_grid_env):
        """Test the loss method when kickstarting is disabled."""
        mock_config.teacher_uri = None

        kickstarter = Kickstarter(mock_config, "cpu", mock_policy_store, mock_metta_grid_env)
        kickstarter.enabled = False

        # Create test tensors
        student_normalized_logits = torch.randn(2, 5)
        student_value = torch.randn(2, 1)
        observation = TensorDict({"obs": torch.randn(2, 10)}, batch_size=[2])

        # Call the loss method
        ks_action_loss, ks_value_loss = kickstarter.loss(500, student_normalized_logits, student_value, observation)

        # Both losses should be zero tensors
        assert torch.all(ks_action_loss == 0.0)
        assert torch.all(ks_value_loss == 0.0)

    @patch("metta.rl.kickstarter.Kickstarter._load_policies")
    def test_loss_after_kickstart_steps(self, mock_load_policies, mock_config, mock_policy_store, mock_metta_grid_env):
        """Test the loss method after kickstart steps have been exceeded."""
        mock_config.teacher_uri = "wandb://teacher/uri"

        kickstarter = Kickstarter(mock_config, "cpu", mock_policy_store, mock_metta_grid_env)
        kickstarter.enabled = True

        # Create test tensors
        student_normalized_logits = torch.randn(2, 5)
        student_value = torch.randn(2, 1)
        observation = TensorDict({"obs": torch.randn(2, 10)}, batch_size=[2])

        # Call the loss method with agent_step > kickstart_steps
        ks_action_loss, ks_value_loss = kickstarter.loss(1500, student_normalized_logits, student_value, observation)

        # Both losses should be zero tensors
        assert torch.all(ks_action_loss == 0.0)
        assert torch.all(ks_value_loss == 0.0)

    @patch("metta.rl.kickstarter.Kickstarter._load_policies")
    def test_loss_with_annealing(self, mock_load_policies, mock_config, mock_policy_store, mock_metta_grid_env):
        """Test the loss method with annealing."""
        mock_config.teacher_uri = "wandb://teacher/uri"

        kickstarter = Kickstarter(mock_config, "cpu", mock_policy_store, mock_metta_grid_env)
        kickstarter.enabled = True

        # Mock the teachers list - need to mock the teacher policy that gets called
        mock_teacher_policy = MagicMock()
        teacher_td = TensorDict(
            {
                "value": torch.ones(2, 1),
                "full_log_probs": torch.log(torch.ones(2, 5) / 5.0),  # Uniform distribution log probs
            },
            batch_size=[2],
        )
        mock_teacher_policy.return_value = teacher_td
        # Create proper teacher config and dictionary structure
        teacher_config = KickstartTeacherConfig(teacher_uri="test://uri", action_loss_coef=0.5, value_loss_coef=0.5)
        kickstarter.teachers = {mock_teacher_policy: teacher_config}

        # Create test tensors
        student_normalized_logits = torch.randn(2, 5)
        student_value = torch.randn(2, 1)
        observation = TensorDict({"obs": torch.randn(2, 10)}, batch_size=[2])

        # Test during ramp down phase (after ramp_down_start_step)
        agent_step = 900  # Between ramp_down_start_step (800) and kickstart_steps (1000)

        # Call the loss method
        ks_action_loss, ks_value_loss = kickstarter.loss(
            agent_step, student_normalized_logits, student_value, observation
        )

        # Check that anneal_factor was updated
        progress = (agent_step - kickstarter.ramp_down_start_step) / kickstarter.anneal_duration
        expected_anneal_factor = 1.0 - progress
        assert kickstarter.anneal_factor == pytest.approx(expected_anneal_factor)

        # Verify that teacher policy was called with the observation
        mock_teacher_policy.assert_called_once_with(observation)

        # Verify that losses are non-zero (scaled by anneal factor)
        assert ks_action_loss != 0.0
        assert ks_value_loss != 0.0
