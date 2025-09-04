"""Tests for PR splitting functionality."""

import json
from unittest.mock import Mock, patch

import pytest

from gitta.split import FileDiff, PRSplitter, SplitDecision


class TestPRSplitter:
    """Test cases for PRSplitter class."""

    @pytest.fixture
    def splitter(self):
        """Create a PRSplitter instance with mocked API key."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            return PRSplitter()

    def test_parse_diff(self, splitter):
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

        files = splitter.parse_diff(diff_text)

        assert len(files) == 2
        assert files[0].filename == "file1.py"
        assert files[1].filename == "file2.py"

        # Check additions/deletions
        assert len(files[0].additions) == 2  # Two lines added
        assert len(files[0].deletions) == 1  # One line removed
        assert len(files[1].additions) == 3  # Three lines in new file

    def test_create_patch_file(self, splitter):
        """Test creating a patch from selected files."""
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

    def test_verify_split(self, splitter):
        """Test verification of split diffs."""
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

        assert splitter.verify_split(original, diff1, diff2) == True

        # Missing line
        diff2_incomplete = """diff --git a/file.py b/file.py"""

        assert splitter.verify_split(original, diff1, diff2_incomplete) == False

    @patch("gitta.split.Anthropic")
    def test_analyze_diff_with_ai(self, mock_anthropic_class, splitter):
        """Test AI analysis of diffs."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock the API response
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text=json.dumps(
                    {
                        "group1_files": ["file1.py"],
                        "group2_files": ["file2.py"],
                        "group1_description": "Backend changes",
                        "group2_description": "Frontend changes",
                        "group1_title": "feat: Update backend logic",
                        "group2_title": "feat: Update frontend UI",
                    }
                )
            )
        ]
        mock_client.messages.create.return_value = mock_response

        # Reinitialize splitter to use mocked client
        splitter.__init__()

        files = [FileDiff("file1.py", ["+line1"], [], [], ""), FileDiff("file2.py", ["+line2"], [], [], "")]

        decision = splitter.analyze_diff_with_ai(files)

        assert isinstance(decision, SplitDecision)
        assert decision.group1_files == ["file1.py"]
        assert decision.group2_files == ["file2.py"]
        assert "backend" in decision.group1_description.lower()
        assert "frontend" in decision.group2_description.lower()

    @patch("gitta.git.get_remote_url")
    def test_get_repo_from_remote(self, mock_get_remote, splitter):
        """Test extracting repo info from remote URL."""
        # Test HTTPS URL
        mock_get_remote.return_value = "https://github.com/owner/repo.git"
        assert splitter.get_repo_from_remote() == "owner/repo"

        # Test SSH URL
        mock_get_remote.return_value = "git@github.com:owner/repo.git"
        assert splitter.get_repo_from_remote() == "owner/repo"

        # Test URL without .git
        mock_get_remote.return_value = "https://github.com/owner/repo"
        assert splitter.get_repo_from_remote() == "owner/repo"

        # Test invalid URL
        mock_get_remote.return_value = "not-a-github-url"
        assert splitter.get_repo_from_remote() is None


@pytest.mark.integration
class TestPRSplitterIntegration:
    """Integration tests that require a real git repository."""

    @pytest.mark.skip(reason="Requires actual git repository")
    def test_full_split_workflow(self):
        """Test the complete PR splitting workflow."""
        # This test would:
        # 1. Create a temporary git repository
        # 2. Make some changes across multiple files
        # 3. Run the splitter
        # 4. Verify two branches are created with correct changes
        # 5. Clean up
        pass
