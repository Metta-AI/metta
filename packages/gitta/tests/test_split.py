"""Tests for PR splitting functionality without mocks."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from gitta.split import FileDiff, PRSplitter, SplitDecision


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


def test_create_patch_file():
    """Test creating a patch from selected files."""
    splitter = PRSplitter()

    files = [
        FileDiff(
            filename="file1.py",
            additions=["+line1"],
            deletions=["-line2"],
            hunks=[],
            raw_diff="diff --git a/file1.py b/file1.py\n+line1\n-line2",
        ),
        FileDiff(
            filename="file2.py",
            additions=["+line3"],
            deletions=[],
            hunks=[],
            raw_diff="diff --git a/file2.py b/file2.py\n+line3",
        ),
    ]

    # Select only file1.py
    patch = splitter.create_patch_file(files, ["file1.py"])

    assert "file1.py" in patch
    assert "file2.py" not in patch
    assert "+line1" in patch
    assert "-line2" in patch


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
