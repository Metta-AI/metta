"""Tests for release continuation and step-skipping logic."""

from datetime import datetime
from unittest.mock import patch

from devops.stable.state import (
    ReleaseState,
    save_state,
)


def test_resolve_version_starts_new_when_no_state(monkeypatch, tmp_path):
    """Test that resolve_version creates a new version when no state exists."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    from devops.stable.release_stable import resolve_version

    version = resolve_version(explicit=None, force_new=False)
    assert version.startswith("2025.")  # Should be current date
    assert len(version.split("-")) == 2  # Should have date and time parts


def test_resolve_version_continues_unreleased_state(monkeypatch, tmp_path):
    """Test that resolve_version continues an existing unreleased version."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create an unreleased state
    state = ReleaseState(
        version="v2025.10.09-1430",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
        released=False,
    )
    save_state(state)

    from devops.stable.release_stable import resolve_version

    version = resolve_version(explicit=None, force_new=False)
    assert version == "2025.10.09-1430"


def test_resolve_version_starts_new_when_previous_released(monkeypatch, tmp_path):
    """Test that resolve_version starts new version when previous is released."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create a released state
    state = ReleaseState(
        version="v2025.10.09-1430",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
        released=True,
    )
    save_state(state)

    from devops.stable.release_stable import resolve_version

    version = resolve_version(explicit=None, force_new=False)
    assert version != "2025.10.09-1430"  # Should be a new version
    assert version.startswith("2025.")


def test_resolve_version_respects_explicit(monkeypatch, tmp_path):
    """Test that resolve_version uses explicit version when provided."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    from devops.stable.release_stable import resolve_version

    version = resolve_version(explicit="2025.01.01-1234", force_new=False)
    assert version == "2025.01.01-1234"


def test_resolve_version_respects_force_new(monkeypatch, tmp_path):
    """Test that resolve_version creates new version when force_new is True."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create an unreleased state
    state = ReleaseState(
        version="v2025.10.09-1430",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
        released=False,
    )
    save_state(state)

    from devops.stable.release_stable import resolve_version

    version = resolve_version(explicit=None, force_new=True)
    assert version != "2025.10.09-1430"  # Should be different
    assert version.startswith("2025.")


def test_step_prepare_tag_skips_when_gate_passed(monkeypatch, tmp_path, capsys):
    """Test that step_prepare_tag skips when gate is already passed."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    from devops.stable.release_stable import step_prepare_tag

    # Create state with completed prepare_tag gate
    state = ReleaseState(
        version="v2025.10.09-1430",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
    )
    state.gates.append(
        {
            "step": "prepare_tag",
            "passed": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Mock git operations so they don't actually run
    with patch("gitta.get_current_commit", return_value="abc123"):
        with patch("gitta.run_git"):
            step_prepare_tag(version="2025.10.09-1430", state=state)

    captured = capsys.readouterr()
    assert "already completed (skipping)" in captured.out
    assert "Creating staging tag" not in captured.out


def test_step_bug_check_skips_when_gate_passed(monkeypatch, tmp_path, capsys):
    """Test that step_bug_check skips when gate is already passed."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    from devops.stable.release_stable import step_bug_check

    # Create state with completed bug_check gate
    state = ReleaseState(
        version="v2025.10.09-1430",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
    )
    state.gates.append(
        {
            "step": "bug_check",
            "passed": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Mock RC commit verification and check_blockers
    with patch("devops.stable.release_stable.verify_on_rc_commit", return_value="abc123"):
        with patch("devops.stable.release_stable.check_blockers"):
            step_bug_check(version="2025.10.09-1430", state=state)

    captured = capsys.readouterr()
    assert "already completed (skipping)" in captured.out


def test_step_prepare_tag_marks_gate_when_complete(monkeypatch, tmp_path):
    """Test that step_prepare_tag marks the gate as passed when complete."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    from devops.stable.release_stable import step_prepare_tag

    # Create state without gates
    state = ReleaseState(
        version="v2025.10.09-1430",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
    )

    # Mock git operations
    with patch("gitta.get_current_commit", return_value="abc123"):
        with patch("gitta.run_git") as mock_git:
            # Mock tag doesn't exist
            mock_git.return_value = ""
            step_prepare_tag(version="2025.10.09-1430", state=state)

    # Verify gate was added
    assert len(state.gates) == 1
    assert state.gates[0]["step"] == "prepare_tag"
    assert state.gates[0]["passed"] is True


def test_step_bug_check_marks_gate_when_complete(monkeypatch, tmp_path):
    """Test that step_bug_check marks the gate as passed when complete."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    from devops.stable.release_stable import step_bug_check

    # Create state without gates
    state = ReleaseState(
        version="v2025.10.09-1430",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
    )

    # Mock RC commit verification and check_blockers to return True (passed)
    with patch("devops.stable.release_stable.verify_on_rc_commit", return_value="abc123"):
        with patch("devops.stable.release_stable.check_blockers", return_value=True):
            step_bug_check(version="2025.10.09-1430", state=state)

    # Verify gate was added
    assert len(state.gates) == 1
    assert state.gates[0]["step"] == "bug_check"
    assert state.gates[0]["passed"] is True


def test_cmd_all_creates_state_once(monkeypatch, tmp_path):
    """Test that cmd_all creates state once and passes it to all steps."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    from devops.stable.release_stable import cmd_all

    # Mock all the step functions
    with patch("devops.stable.release_stable.step_prepare_tag") as mock_prepare:
        with patch("devops.stable.release_stable.step_bug_check") as mock_bug:
            with patch("devops.stable.release_stable.step_task_validation") as mock_task:
                with patch("devops.stable.release_stable.step_summary") as mock_summary:
                    with patch("devops.stable.release_stable._VERSION", "2025.10.09-1430"):
                        with patch("gitta.get_current_commit", return_value="abc123"):
                            cmd_all(task=None, reeval=False)

    # Verify all steps were called
    assert mock_prepare.called
    assert mock_bug.called
    assert mock_task.called
    assert mock_summary.called

    # Verify they all received the same state object
    prepare_state = mock_prepare.call_args.kwargs["state"]
    bug_state = mock_bug.call_args.kwargs["state"]
    assert prepare_state is bug_state


def test_continuation_skips_completed_steps(monkeypatch, tmp_path, capsys):
    """Test that steps with completed gates are skipped."""
    monkeypatch.setattr("devops.stable.state.get_repo_root", lambda: tmp_path)

    # Create a state with completed prepare_tag and bug_check gates
    state = ReleaseState(
        version="v2025.10.09-1430",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
        released=False,
    )
    state.gates.append({"step": "prepare_tag", "passed": True, "timestamp": datetime.utcnow().isoformat()})
    state.gates.append({"step": "bug_check", "passed": True, "timestamp": datetime.utcnow().isoformat()})
    save_state(state)

    from devops.stable.release_stable import step_bug_check, step_prepare_tag

    # Mock git operations
    with patch("gitta.get_current_commit", return_value="abc123"):
        with patch("gitta.run_git", return_value="abc123"):
            # Run the steps - they should skip since gates are marked complete
            step_prepare_tag(version="2025.10.09-1430", state=state)
            step_bug_check(version="2025.10.09-1430", state=state)

    # Verify both steps reported they were skipped
    output = capsys.readouterr().out
    assert output.count("already completed (skipping)") == 2
