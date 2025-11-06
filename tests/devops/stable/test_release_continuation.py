"""Tests for release continuation and step-skipping logic."""

import datetime
import unittest.mock

import devops.stable.state


def test_resolve_version_starts_new_when_no_state(monkeypatch, tmp_path):
    """Test that resolve_version creates a new version when no state exists."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    import devops.stable.release_stable

    version = devops.stable.release_stable.resolve_version(explicit=None, force_new=False)
    assert version.startswith("2025.")  # Should be current date
    assert len(version.split("-")) == 2  # Should have date and time parts


def test_resolve_version_continues_unreleased_state(monkeypatch, tmp_path):
    """Test that resolve_version continues an existing unreleased version."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create an unreleased state
    state = devops.stable.state.ReleaseState(
        version="v2025.10.09-143000",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        commit_sha="abc123",
        released=False,
    )
    devops.stable.state.save_state(state)

    version = devops.stable.release_stable.resolve_version(explicit=None, force_new=False)
    assert version == "2025.10.09-143000"


def test_resolve_version_starts_new_when_previous_released(monkeypatch, tmp_path):
    """Test that resolve_version starts new version when previous is released."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create a released state
    state = devops.stable.state.ReleaseState(
        version="v2025.10.09-143000",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        commit_sha="abc123",
        released=True,
    )
    devops.stable.state.save_state(state)

    version = devops.stable.release_stable.resolve_version(explicit=None, force_new=False)
    assert version != "2025.10.09-143000"  # Should be a new version
    assert version.startswith("2025.")


def test_resolve_version_respects_explicit(monkeypatch, tmp_path):
    """Test that resolve_version uses explicit version when provided."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    version = devops.stable.release_stable.resolve_version(explicit="2025.01.01-1234", force_new=False)
    assert version == "2025.01.01-1234"


def test_resolve_version_respects_force_new(monkeypatch, tmp_path):
    """Test that resolve_version creates new version when force_new is True."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create an unreleased state
    state = devops.stable.state.ReleaseState(
        version="v2025.10.09-143000",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        commit_sha="abc123",
        released=False,
    )
    devops.stable.state.save_state(state)

    version = devops.stable.release_stable.resolve_version(explicit=None, force_new=True)
    assert version != "2025.10.09-143000"  # Should be different
    assert version.startswith("2025.")


def test_step_prepare_tag_skips_when_gate_passed(monkeypatch, tmp_path, capsys):
    """Test that step_prepare_tag skips when gate is already passed."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create state with completed prepare_tag gate
    state = devops.stable.state.ReleaseState(
        version="v2025.10.09-143000",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        commit_sha="abc123",
    )
    state.gates.append(
        devops.stable.state.Gate(
            step="prepare_tag",
            passed=True,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        )
    )

    # Mock git operations so they don't actually run
    with unittest.mock.patch("gitta.get_current_commit", return_value="abc123"):
        with unittest.mock.patch("gitta.run_git"):
            devops.stable.release_stable.step_prepare_tag(version="2025.10.09-143000", state=state)

    captured = capsys.readouterr()
    assert "Step already completed" in captured.out
    assert "Creating staging tag" not in captured.out


def test_step_bug_check_skips_when_gate_passed(monkeypatch, tmp_path, capsys):
    """Test that step_bug_check skips when gate is already passed."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create state with completed bug_check gate
    state = devops.stable.state.ReleaseState(
        version="v2025.10.09-143000",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        commit_sha="abc123",
    )
    state.gates.append(
        devops.stable.state.Gate(
            step="bug_check",
            passed=True,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        )
    )

    # Mock RC commit verification and check_blockers
    with unittest.mock.patch("devops.stable.release_stable.verify_on_rc_commit", return_value="abc123"):
        with unittest.mock.patch("devops.stable.release_stable.check_blockers"):
            devops.stable.release_stable.step_bug_check(version="2025.10.09-143000", state=state)

    captured = capsys.readouterr()
    assert "Step already completed" in captured.out


def test_step_prepare_tag_marks_gate_when_complete(monkeypatch, tmp_path):
    """Test that step_prepare_tag marks the gate as passed when complete."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create state without gates
    state = devops.stable.state.ReleaseState(
        version="v2025.10.09-143000",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        commit_sha="abc123",
    )

    # Mock git operations
    with unittest.mock.patch("gitta.get_current_commit", return_value="abc123"):
        with unittest.mock.patch("gitta.run_git") as mock_git:
            # Mock tag doesn't exist
            mock_git.return_value = ""
            devops.stable.release_stable.step_prepare_tag(version="2025.10.09-143000", state=state)

    # Verify gate was added
    assert len(state.gates) == 1
    assert state.gates[0].step == "prepare_tag"
    assert state.gates[0].passed is True


def test_step_bug_check_marks_gate_when_complete(monkeypatch, tmp_path):
    """Test that step_bug_check marks the gate as passed when complete."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create state without gates
    state = devops.stable.state.ReleaseState(
        version="v2025.10.09-143000",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        commit_sha="abc123",
    )

    # Mock RC commit verification and check_blockers to return True (passed)
    with unittest.mock.patch("devops.stable.release_stable.verify_on_rc_commit", return_value="abc123"):
        with unittest.mock.patch("devops.stable.release_stable.check_blockers", return_value=True):
            devops.stable.release_stable.step_bug_check(version="2025.10.09-143000", state=state)

    # Verify gate was added
    assert len(state.gates) == 1
    assert state.gates[0].step == "bug_check"
    assert state.gates[0].passed is True


def test_cmd_validate_runs_validation_pipeline(monkeypatch, tmp_path):
    """Test that cmd_validate runs the full validation pipeline."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create mock context with version and skip_commit_match
    mock_ctx = unittest.mock.MagicMock()
    mock_ctx.obj = {
        "version": "2025.10.09-143000",
        "skip_commit_match": False,
    }

    # Mock all the step functions
    with unittest.mock.patch("devops.stable.release_stable.step_prepare_tag") as mock_prepare:
        with unittest.mock.patch("devops.stable.release_stable.step_job_validation") as mock_job:
            with unittest.mock.patch("devops.stable.release_stable.step_summary") as mock_summary:
                with unittest.mock.patch("gitta.get_current_commit", return_value="abc123"):
                    devops.stable.release_stable.cmd_validate(ctx=mock_ctx, job=None, retry=False)

    # Verify validation pipeline steps were called
    assert mock_prepare.called
    assert mock_job.called
    assert mock_summary.called

    # Verify they all received the same state object
    prepare_state = mock_prepare.call_args.kwargs["state"]
    assert prepare_state is not None


def test_cmd_hotfix_skips_validation(monkeypatch, tmp_path):
    """Test that cmd_hotfix skips validation step."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create mock context with version and skip_commit_match
    mock_ctx = unittest.mock.MagicMock()
    mock_ctx.obj = {
        "version": "2025.10.09-143000",
        "skip_commit_match": False,
    }

    # Mock all the step functions
    with unittest.mock.patch("devops.stable.release_stable.step_prepare_tag") as mock_prepare:
        with unittest.mock.patch("devops.stable.release_stable.step_job_validation") as mock_job:
            with unittest.mock.patch("devops.stable.release_stable.step_release") as mock_release:
                with unittest.mock.patch("gitta.get_current_commit", return_value="abc123"):
                    devops.stable.release_stable.cmd_hotfix(ctx=mock_ctx)

    # Verify hotfix pipeline: prepare-tag and release, NO validation
    assert mock_prepare.called
    assert not mock_job.called  # Validation should be skipped
    assert mock_release.called


def test_cmd_release_runs_bug_check_and_release(monkeypatch, tmp_path):
    """Test that cmd_release runs bug check then creates release."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create mock context with version and skip_commit_match
    mock_ctx = unittest.mock.MagicMock()
    mock_ctx.obj = {
        "version": "2025.10.09-143000",
        "skip_commit_match": False,
    }

    # Mock the step functions
    with unittest.mock.patch("devops.stable.release_stable.step_bug_check") as mock_bug:
        with unittest.mock.patch("devops.stable.release_stable.step_release") as mock_release:
            with unittest.mock.patch("gitta.get_current_commit", return_value="abc123"):
                devops.stable.release_stable.cmd_release(ctx=mock_ctx)

    # Verify release pipeline: bug check then release
    assert mock_bug.called
    assert mock_release.called

    # Verify they received state
    bug_state = mock_bug.call_args.kwargs["state"]
    assert bug_state is not None


def test_continuation_skips_completed_steps(monkeypatch, tmp_path, capsys):
    """Test that steps with completed gates are skipped."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create a state with completed prepare_tag and bug_check gates
    state = devops.stable.state.ReleaseState(
        version="v2025.10.09-143000",
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        commit_sha="abc123",
        released=False,
    )
    state.gates.append(
        devops.stable.state.Gate(
            step="prepare_tag", passed=True, timestamp=datetime.datetime.now(datetime.UTC).isoformat()
        )
    )
    state.gates.append(
        devops.stable.state.Gate(
            step="bug_check", passed=True, timestamp=datetime.datetime.now(datetime.UTC).isoformat()
        )
    )
    devops.stable.state.save_state(state)

    # Mock git operations
    with unittest.mock.patch("gitta.get_current_commit", return_value="abc123"):
        with unittest.mock.patch("gitta.run_git", return_value="abc123"):
            # Run the steps - they should skip since gates are marked complete
            devops.stable.release_stable.step_prepare_tag(version="2025.10.09-143000", state=state)
            devops.stable.release_stable.step_bug_check(version="2025.10.09-143000", state=state)

    # Verify both steps reported they were skipped
    output = capsys.readouterr().out
    assert output.count("Step already completed") == 2
