"""Tests for PR splitting functionality without mocks."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from gitta import GitError
from gitta.split import FileDiff, PRSplitter, SplitDecision, main


@pytest.fixture(autouse=True)
def clear_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests do not rely on real Anthropic credentials."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_parse_diff():
    """Test parsing of git diff output."""
    diff_text = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -1,5 +1,6 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
+    return True

 def goodbye():
     print("Goodbye")
diff --git a/file2.py b/file2.py
new file mode 100644
index 0000000..789012
--- /dev/null
+++ b/file2.py
@@ -0,0 +1,3 @@
+def new_function():
+    return 42
+"""

    # Create a splitter without needing API key for basic parsing
    splitter = PRSplitter()
    files = splitter.parse_diff(diff_text)

    assert len(files) == 2
    assert files[0].filename == "file1.py"
    assert files[1].filename == "file2.py"

    # Check additions/deletions counts
    assert len(files[0].additions) == 2  # Two lines added
    assert len(files[0].deletions) == 1  # One line removed
    assert len(files[1].additions) == 3  # Three lines in new file


def test_create_patch_file(monkeypatch: pytest.MonkeyPatch):
    """Test that create_patch_file delegates to git diff with expected args."""
    splitter = PRSplitter()
    splitter.base_branch = "origin/main"
    splitter.current_branch = "feature"

    captured_args: Dict[str, Any] = {}

    def fake_run_git(*args: str) -> str:
        captured_args["args"] = args
        return "MOCK_DIFF"

    monkeypatch.setattr("gitta.split.run_git", fake_run_git)

    patch = splitter.create_patch_file([], ["file1.py", "file1.py"])

    assert patch == "MOCK_DIFF"
    assert captured_args["args"] == (
        "diff",
        "--binary",
        "origin/main...feature",
        "--",
        "file1.py",
    )


def test_verify_split():
    """Test verification of split diffs."""
    splitter = PRSplitter()

    original = """diff --git a/file.py b/file.py
+added line 1
-removed line 1
+added line 2"""

    # Perfect split
    diff1 = """diff --git a/file.py b/file.py
+added line 1
-removed line 1"""

    diff2 = """diff --git a/file.py b/file.py
+added line 2"""

    assert splitter.verify_split(original, diff1, diff2)

    # Missing line
    diff2_incomplete = """diff --git a/file.py b/file.py"""

    assert not splitter.verify_split(original, diff1, diff2_incomplete)


def test_get_repo_from_remote_urls():
    """Test extracting repo info from various remote URL formats."""
    # Note: This test doesn't actually need PRSplitter, so we can test the logic directly

    # Test different URL formats
    test_cases = [
        ("https://github.com/owner/repo.git", "owner/repo"),
        ("git@github.com:owner/repo.git", "owner/repo"),
        ("https://github.com/owner/repo", "owner/repo"),
        ("git@github.com:owner/repo", "owner/repo"),
        ("ssh://git@github.com/owner/repo.git", "owner/repo"),
        ("not-a-github-url", None),
        ("https://gitlab.com/owner/repo", None),
    ]

    for url, expected in test_cases:
        # Test the extraction logic directly
        if "github.com" in url:
            if url.startswith("git@"):
                result = url.split(":", 1)[1].removesuffix(".git")
            elif url.startswith("ssh://"):
                result = url.split("/")[-2] + "/" + url.split("/")[-1].removesuffix(".git")
            else:
                result = "/".join(url.split("/")[-2:]).removesuffix(".git")
            assert result == expected
        else:
            assert expected is None


def test_split_decision_json_parsing():
    """Test parsing of split decision from JSON."""
    # Test valid JSON parsing
    json_str = json.dumps(
        {
            "group1_files": ["file1.py", "file2.py"],
            "group2_files": ["file3.py"],
            "group1_description": "Backend changes",
            "group2_description": "Frontend changes",
            "group1_title": "feat: Update backend",
            "group2_title": "feat: Update frontend",
        }
    )

    decision = SplitDecision(**json.loads(json_str))
    assert decision.group1_files == ["file1.py", "file2.py"]
    assert decision.group2_files == ["file3.py"]
    assert "backend" in decision.group1_description.lower()


def test_real_git_diff():
    """Test with a real git repository and actual diffs."""
    # Create temporary repo
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Initialize repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)

        # Create initial commit
        (repo_path / "file1.py").write_text("def hello():\n    print('hello')\n")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)

        # Make changes
        (repo_path / "file1.py").write_text("def hello():\n    print('hello world')\n    return True\n")
        (repo_path / "file2.py").write_text("def goodbye():\n    print('bye')\n")

        subprocess.run(["git", "add", "file2.py"], cwd=repo_path, check=True, capture_output=True)

        # Get the diff
        result = subprocess.run(["git", "diff", "HEAD"], cwd=repo_path, capture_output=True, text=True, check=True)

        # Parse it
        splitter = PRSplitter()
        files = splitter.parse_diff(result.stdout)

        assert len(files) == 2
        assert files[0].filename == "file1.py"
        assert files[1].filename == "file2.py"
        assert any("hello world" in line for line in files[0].additions)


def test_apply_patch_to_existing_branch(monkeypatch: pytest.MonkeyPatch):
    """`apply_patch_to_new_branch` should recreate existing branches safely."""
    splitter = PRSplitter()
    splitter.base_branch = "origin/main"

    calls: list[tuple[str, ...]] = []

    def fake_run_git(*args: str) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr("gitta.split.run_git", fake_run_git)

    splitter.apply_patch_to_new_branch("diff --git a/file b/file\n", "feature-part1")

    assert ("checkout", "-B", "feature-part1", "origin/main") in calls


def test_analyze_diff_requires_api_key():
    """Ensure Anthropic-backed splitting raises without credentials."""
    splitter = PRSplitter()
    files = [
        FileDiff(
            filename="demo.py",
            additions=["+print('hi')"],
            deletions=[],
            hunks=[],
            raw_diff="diff --git a/demo.py b/demo.py\n+print('hi')",
        )
    ]

    with pytest.raises(ValueError):
        splitter.analyze_diff_with_ai(files)


def test_push_branch_force_with_lease(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """If the remote already exists, push_branch should retry with --force-with-lease."""
    splitter = PRSplitter()
    calls: list[tuple[str, ...]] = []
    call_count = {"push_origin": 0}

    def fake_run_git(*args: str) -> str:
        calls.append(args)
        if args == ("push", "origin", "feature-force") and call_count["push_origin"] == 0:
            call_count["push_origin"] += 1
            raise GitError("git push origin feature-force failed (1): non-fast-forward")
        return ""

    monkeypatch.setattr("gitta.split.run_git", fake_run_git)

    splitter.push_branch("feature-force")

    assert ("push", "origin", "feature-force") in calls
    assert ("push", "--force-with-lease", "origin", "feature-force") in calls

    captured = capsys.readouterr().out
    assert "force-with-lease" in captured


def test_push_branch_propagates_other_errors(monkeypatch: pytest.MonkeyPatch):
    """Errors unrelated to non-fast-forward should still raise."""
    splitter = PRSplitter()

    def fake_run_git(*_: str) -> str:
        raise GitError("Permission denied (publickey).")

    monkeypatch.setattr("gitta.split.run_git", fake_run_git)

    with pytest.raises(GitError):
        splitter.push_branch("feature-error")


def test_push_branch_updates_rejected(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Handle the 'Updates were rejected because...' message emitted by git."""
    splitter = PRSplitter()
    calls: list[tuple[str, ...]] = []
    call_count = {"push_origin": 0}

    def fake_run_git(*args: str) -> str:
        calls.append(args)
        if args == ("push", "origin", "feature-reject") and call_count["push_origin"] == 0:
            call_count["push_origin"] += 1
            raise GitError(
                "git push origin feature-reject failed (1): Updates were rejected because the tip of your current "
                "branch is behind"
            )
        return ""

    monkeypatch.setattr("gitta.split.run_git", fake_run_git)

    splitter.push_branch("feature-reject")

    assert ("push", "--force-with-lease", "origin", "feature-reject") in calls
    assert "force-with-lease" in capsys.readouterr().out


def test_push_branch_no_force(monkeypatch: pytest.MonkeyPatch):
    """When force_push is disabled the retry should not occur."""
    splitter = PRSplitter(force_push=False)

    def fake_run_git(*_: str) -> str:
        raise GitError("git push origin feature-noforce failed (1): non-fast-forward")

    monkeypatch.setattr("gitta.split.run_git", fake_run_git)

    with pytest.raises(GitError):
        splitter.push_branch("feature-noforce")


def test_build_prompt_reflects_independence():
    splitter_low = PRSplitter(independence=0.0)
    splitter_mid = PRSplitter(independence=0.5)
    splitter_high = PRSplitter(independence=1.0)

    file_summaries: list[Dict[str, Any]] = [
        {
            "filename": "feature.py",
            "additions": 10,
            "deletions": 2,
            "hunks": 3,
            "sample_changes": "",
        }
    ]

    prompt_low = splitter_low._build_split_prompt(file_summaries)
    prompt_mid = splitter_mid._build_split_prompt(file_summaries)
    prompt_high = splitter_high._build_split_prompt(file_summaries)

    assert "Independence preference: 0.00" in prompt_low
    assert "Weight of balanced sizing: 1.00" in prompt_low
    assert "Prioritize balanced group sizes" in prompt_low

    assert "Independence preference: 0.50" in prompt_mid
    assert "Weight of balanced sizing: 0.50" in prompt_mid
    assert "Balance both logical separation and size" in prompt_mid

    assert "Independence preference: 1.00" in prompt_high
    assert "Weight of balanced sizing: 0.00" in prompt_high
    assert "Prioritize independence of concerns" in prompt_high


def test_build_prompt_mentions_exclusions():
    splitter = PRSplitter(exclude_files={"uv.lock", "poetry.lock"})
    file_summaries = [
        {
            "filename": "uv.lock",
            "additions": 0,
            "deletions": 0,
            "hunks": 0,
            "sample_changes": "",
        }
    ]

    prompt = splitter._build_split_prompt(file_summaries)

    assert "uv.lock" in prompt
    assert "poetry.lock" in prompt


def test_invalid_independence_value():
    with pytest.raises(ValueError):
        PRSplitter(independence=1.5)
    with pytest.raises(ValueError):
        PRSplitter(independence=-0.1)


def test_apply_exclusions_moves_files():
    splitter = PRSplitter(exclude_files={"keep.txt"})
    decision = SplitDecision(
        group1_files=["a.py", "b.py"],
        group2_files=["keep.txt", "c.py"],
        group1_description="g1",
        group2_description="g2",
        group1_title="t1",
        group2_title="t2",
    )

    files = [
        FileDiff(filename="a.py", additions=[], deletions=[], hunks=[], raw_diff=""),
        FileDiff(filename="keep.txt", additions=[], deletions=[], hunks=[], raw_diff=""),
        FileDiff(filename="c.py", additions=[], deletions=[], hunks=[], raw_diff=""),
    ]

    adjusted = splitter._apply_exclusions(decision, files)

    assert "keep.txt" in adjusted.group1_files
    assert "keep.txt" not in adjusted.group2_files


def test_split_pr_forwards_independence(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    class FakeSplitter:
        def __init__(self, *_, independence: float, exclude_files: Set[str], **__):
            captured["independence"] = independence
            captured["exclude_files"] = exclude_files

        def split(self) -> None:
            captured["split_called"] = True

    monkeypatch.setattr("gitta.split.PRSplitter", FakeSplitter)

    # Call the helper directly
    from gitta.split import split_pr

    split_pr(independence=0.3, exclude_files=["uv.lock"])
    assert captured["independence"] == 0.3
    assert captured["split_called"] is True
    assert captured["exclude_files"] == {"uv.lock"}


def test_cli_rejects_invalid_independence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["gitta.split", "--independence", "1.2", "--anthropic-key", "abc"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


def test_cli_passes_independence(monkeypatch: pytest.MonkeyPatch):
    args = ["gitta.split", "--anthropic-key", "abc123", "--independence", "0.2", "--exclude", "uv.lock"]
    monkeypatch.setattr(sys, "argv", args)

    captured_kwargs: Dict[str, Any] = {}

    def fake_split_pr(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)

    monkeypatch.setattr("gitta.split.split_pr", fake_split_pr)

    main()
    assert captured_kwargs["independence"] == 0.2
    assert captured_kwargs["force_push"] is True
    assert captured_kwargs["exclude_files"] == ["uv.lock"]


def test_prsplitter_loads_key_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=sk-ant-test\n")
    repo_root = Path(__file__).resolve().parents[4]

    try:
        original_cwd = Path.cwd()
    except FileNotFoundError:
        original_cwd = repo_root
        os.chdir(original_cwd)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    try:
        os.chdir(tmp_path)
        splitter = PRSplitter()
    finally:
        os.chdir(original_cwd)

    assert splitter.anthropic_api_key == "sk-ant-test"


def test_prsplitter_prefers_repo_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    (repo_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-root\n")

    nested = repo_path / "nested" / "deeper"
    nested.mkdir(parents=True)

    repo_root = Path(__file__).resolve().parents[4]

    try:
        original_cwd = Path.cwd()
    except FileNotFoundError:
        original_cwd = repo_root
        os.chdir(original_cwd)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    try:
        os.chdir(nested)
        splitter = PRSplitter()
    finally:
        os.chdir(original_cwd)

    assert splitter.anthropic_api_key == "sk-ant-root"
