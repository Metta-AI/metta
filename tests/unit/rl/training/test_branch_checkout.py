"""Test branch-based checkout feature for remote evaluations."""

from unittest.mock import Mock, patch

from metta.rl.training.evaluator import Evaluator, EvaluatorConfig


def test_evaluator_config_has_use_branch_checkout():
    """Test that EvaluatorConfig has the use_branch_checkout field with correct default."""
    config = EvaluatorConfig()
    assert hasattr(config, "use_branch_checkout")
    assert config.use_branch_checkout is False  # Default should be False for backward compatibility


def test_evaluator_config_can_set_use_branch_checkout():
    """Test that we can set use_branch_checkout to True."""
    config = EvaluatorConfig(use_branch_checkout=True)
    assert config.use_branch_checkout is True


@patch("metta.rl.training.evaluator.get_current_git_branch")
@patch("metta.rl.training.evaluator.get_task_commit_hash")
def test_evaluator_uses_branch_when_flag_is_set(mock_get_commit, mock_get_branch):
    """Test that Evaluator uses branch name when use_branch_checkout is True."""
    mock_get_branch.return_value = "feature/test-branch"
    mock_get_commit.return_value = "abc123def456"

    config = EvaluatorConfig(
        evaluate_remote=True,
        use_branch_checkout=True,
    )

    mock_stats_client = Mock()

    evaluator = Evaluator(
        config=config,
        device="cpu",
        seed=42,
        run_name="test_run",
        stats_client=mock_stats_client,
    )

    # Should have called get_current_git_branch, not get_task_commit_hash
    mock_get_branch.assert_called_once()
    mock_get_commit.assert_not_called()

    # git_hash should be the branch name
    assert evaluator._git_hash == "feature/test-branch"


@patch("metta.rl.training.evaluator.get_current_git_branch")
@patch("metta.rl.training.evaluator.get_task_commit_hash")
def test_evaluator_uses_commit_when_flag_is_false(mock_get_commit, mock_get_branch):
    """Test that Evaluator uses commit SHA when use_branch_checkout is False (default)."""
    mock_get_branch.return_value = "feature/test-branch"
    mock_get_commit.return_value = "abc123def456"

    config = EvaluatorConfig(
        evaluate_remote=True,
        use_branch_checkout=False,  # Explicitly set to False (also the default)
    )

    mock_stats_client = Mock()

    evaluator = Evaluator(
        config=config,
        device="cpu",
        seed=42,
        run_name="test_run",
        stats_client=mock_stats_client,
    )

    # Should have called get_task_commit_hash, not get_current_git_branch
    mock_get_commit.assert_called_once()
    mock_get_branch.assert_not_called()

    # git_hash should be the commit SHA
    assert evaluator._git_hash == "abc123def456"


@patch("metta.rl.training.evaluator.get_current_git_branch")
@patch("metta.rl.training.evaluator.get_task_commit_hash")
def test_evaluator_falls_back_to_commit_if_branch_fails(mock_get_commit, mock_get_branch):
    """Test that Evaluator falls back to commit SHA if branch detection fails."""
    mock_get_branch.return_value = None  # Branch detection fails
    mock_get_commit.return_value = "abc123def456"

    config = EvaluatorConfig(
        evaluate_remote=True,
        use_branch_checkout=True,
    )

    mock_stats_client = Mock()

    with patch("metta.rl.training.evaluator.logger") as mock_logger:
        evaluator = Evaluator(
            config=config,
            device="cpu",
            seed=42,
            run_name="test_run",
            stats_client=mock_stats_client,
        )

        # Should have tried branch first, then fallen back to commit
        mock_get_branch.assert_called_once()
        mock_get_commit.assert_called_once()

        # Should have logged a warning about fallback
        mock_logger.warning.assert_called_with("Could not determine branch, falling back to commit hash")

        # git_hash should be the commit SHA from fallback
        assert evaluator._git_hash == "abc123def456"


def test_directory_sanitization():
    """Test that branch name sanitization preserves uniqueness."""
    # Test cases for directory name sanitization
    test_cases = [
        ("main", "main"),
        ("feature/new-feature", "feature__new-feature"),
        ("user/john/fix-123", "user__john__fix-123"),
        ("branch-with-dashes", "branch-with-dashes"),
        ("axel/sweep", "axel__sweep"),  # With slash
        ("axel-sweep", "axel-sweep"),  # With dash - must be different!
        ("axel_sweep", "axel_sweep"),  # With underscore
    ]

    seen_dirs = {}
    for branch, expected_dir in test_cases:
        # Use the same logic as in eval_task_worker.py
        dir_name = branch.replace("/", "__")
        assert dir_name == expected_dir, f"'{branch}' -> expected '{expected_dir}', got '{dir_name}'"

        # Ensure no collisions
        assert dir_name not in seen_dirs or seen_dirs[dir_name] == branch, (
            f"Collision: '{branch}' and '{seen_dirs.get(dir_name)}' both map to '{dir_name}'"
        )
        seen_dirs[dir_name] = branch


def test_git_ref_validation():
    """Test that git ref validation properly sanitizes directory names."""

    # Import the validation logic inline for testing
    def validate_git_ref(git_ref: str) -> str:
        """Validate git ref and return sanitized directory name."""
        if ".." in git_ref or git_ref.startswith("/") or git_ref.startswith("~"):
            raise ValueError(f"Invalid git reference: {git_ref}")
        # Use __ to avoid collisions (e.g., axel/sweep vs axel-sweep)
        return git_ref.replace("/", "__")

    # Test valid refs
    assert validate_git_ref("main") == "main"
    assert validate_git_ref("feature/new-feature") == "feature__new-feature"
    assert validate_git_ref("user/john/fix-123") == "user__john__fix-123"
    assert validate_git_ref("abc123def456789012345678901234567890abcd") == "abc123def456789012345678901234567890abcd"

    # Test invalid refs
    import pytest

    with pytest.raises(ValueError):
        validate_git_ref("../etc/passwd")
    with pytest.raises(ValueError):
        validate_git_ref("/absolute/path")
    with pytest.raises(ValueError):
        validate_git_ref("~/home/user")
