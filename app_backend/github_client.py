import logging
import subprocess
from typing import Any, Dict, List, Tuple

import httpx

logger = logging.getLogger("github_client")


def run_gh(*args: str) -> str:
    """Run a GitHub CLI command and return its output."""
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        from app_backend.git_client import GitError

        raise GitError(f"GitHub CLI command failed ({e.returncode}): {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        from app_backend.git_client import GitError

        raise GitError("GitHub CLI (gh) is not installed!") from e


class GitCommit:
    """Represents a git commit with relevant information."""

    def __init__(self, hash: str, message: str, author: str, date: str):
        self.hash = hash
        self.message = message
        self.author = author
        self.date = date

    def __repr__(self) -> str:
        return f"GitCommit(hash='{self.hash[:8]}', message='{self.message[:50]}...')"


class GitHubClient:
    """Client for GitHub operations using only HTTP requests."""

    REPO_OWNER = "Metta-AI"
    REPO_NAME = "metta"

    def __init__(self, http_client: httpx.Client):
        """
        Initialize the GitHub client.

        Args:
            http_client: HTTP client implementation to use for requests
        """
        self.http_client = http_client
        self._compare_cache: Dict[str, Dict] = {}  # Cache for compare API responses

    def _get_compare_data(self, base: str, head: str) -> Dict:
        """
        Get compare data from GitHub API with caching.

        Args:
            base: Base branch or commit
            head: Head commit

        Returns:
            GitHub API compare response data
        """
        cache_key = f"{base}...{head}"

        if cache_key in self._compare_cache:
            logger.debug(f"GitHub API: Using cached compare data for {cache_key}")
            return self._compare_cache[cache_key]

        logger.debug(f"GitHub API: Fetching compare data for {cache_key}")
        url = f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}/compare/{cache_key}"
        response = self.http_client.get(url)
        response.raise_for_status()

        data = response.json()
        self._compare_cache[cache_key] = data
        return data

    def get_merge_base(self, commit_hash: str, base_branch: str = "main") -> str:
        """
        Get merge base between commit and base branch using GitHub API.

        Args:
            commit_hash: The commit hash
            base_branch: The base branch (default: main)

        Returns:
            The merge base commit hash

        Raises:
            httpx.HTTPStatusError: If GitHub API request fails
        """
        logger.debug(f"GitHub API: Getting merge base for {commit_hash[:8]} from {base_branch}")
        data = self._get_compare_data(base_branch, commit_hash)
        merge_base = data["merge_base_commit"]["sha"]
        logger.debug(f"GitHub API: Found merge base {merge_base[:8]}")
        return merge_base

    def get_commit_range(self, merge_base: str, commit_hash: str) -> List[GitCommit]:
        """
        Get commits between merge base and target commit using GitHub API.

        Args:
            merge_base: The merge base commit hash
            commit_hash: The target commit hash

        Returns:
            List of GitCommit objects

        Raises:
            httpx.HTTPStatusError: If GitHub API request fails
        """
        logger.debug(f"GitHub API: Getting commit range {merge_base[:8]}...{commit_hash[:8]}")
        data = self._get_compare_data(merge_base, commit_hash)
        commits = []

        for commit_data in data["commits"]:
            commits.append(
                GitCommit(
                    hash=commit_data["sha"],
                    message=commit_data["commit"]["message"],
                    author=commit_data["commit"]["author"]["name"],
                    date=commit_data["commit"]["author"]["date"][:10],  # Extract date only
                )
            )

        logger.debug(f"GitHub API: Retrieved {len(commits)} commits")
        return commits

    def get_range_diff_stats(self, merge_base: str, commit_hash: str) -> str:
        """
        Get the diff stats for the range between merge base and commit using GitHub API.

        Args:
            merge_base: The merge base commit hash
            commit_hash: The target commit hash

        Returns:
            Diff summary string with file stats

        Raises:
            httpx.HTTPStatusError: If GitHub API request fails
        """
        logger.debug(f"GitHub API: Getting range diff stats {merge_base[:8]}...{commit_hash[:8]}")
        data = self._get_compare_data(merge_base, commit_hash)

        # Extract file stats from the comparison
        files_changed = len(data.get("files", []))
        total_additions = sum(f.get("additions", 0) for f in data.get("files", []))
        total_deletions = sum(f.get("deletions", 0) for f in data.get("files", []))

        # Format diff stats similar to git diff --stat
        diff_stats = [f" {files_changed} files changed"]
        if total_additions > 0:
            diff_stats.append(f"{total_additions} insertions(+)")
        if total_deletions > 0:
            diff_stats.append(f"{total_deletions} deletions(-)")

        # Add file-by-file breakdown
        file_stats = []
        for file_info in data.get("files", []):
            filename = file_info.get("filename", "")
            additions = file_info.get("additions", 0)
            deletions = file_info.get("deletions", 0)
            changes = file_info.get("changes", 0)

            # Format like git diff --stat
            file_line = f" {filename}"
            if changes > 0:
                file_line += f" | {changes} "
                if additions > 0:
                    file_line += "+" * min(additions, 20)
                if deletions > 0:
                    file_line += "-" * min(deletions, 20)
            file_stats.append(file_line)

        result = "\n".join(file_stats)
        if diff_stats:
            result += "\n " + ", ".join(diff_stats)

        logger.debug(f"GitHub API: Retrieved diff stats for {files_changed} files")
        return result

    def get_range_diff(self, merge_base: str, commit_hash: str) -> str:
        """
        Get the actual diff content for the range between merge base and commit using GitHub API.

        Args:
            merge_base: The merge base commit hash
            commit_hash: The target commit hash

        Returns:
            Actual diff content with + and - lines

        Raises:
            httpx.HTTPStatusError: If GitHub API request fails
        """
        logger.debug(f"GitHub API: Getting range diff {merge_base[:8]}...{commit_hash[:8]}")
        data = self._get_compare_data(merge_base, commit_hash)

        # Build actual diff content from file patches
        diff_content = []
        for file_info in data.get("files", []):
            filename = file_info.get("filename", "")
            patch = file_info.get("patch", "")

            if patch:
                # Add file header
                diff_content.append(f"diff --git a/{filename} b/{filename}")
                diff_content.append(f"--- a/{filename}")
                diff_content.append(f"+++ b/{filename}")
                diff_content.append(patch)
                diff_content.append("")  # Empty line between files

        result = "\n".join(diff_content)
        logger.debug(f"GitHub API: Retrieved actual diff content for {len(data.get('files', []))} files")
        return result

    def get_all_commit_data(self, commit_hash: str, base_branch: str = "main") -> Tuple[str, List[GitCommit], str, str]:
        """
        Get all commit data (merge base, commits, file stats, diff) in a single optimized call.

        Args:
            commit_hash: The target commit hash
            base_branch: The base branch (default: main)

        Returns:
            Tuple of (merge_base, commits, file_stats, actual_diff)
        """
        logger.debug(f"GitHub API: Getting all commit data for {commit_hash[:8]} from {base_branch}")

        # Get API data
        merge_base_data = self._get_compare_data(base_branch, commit_hash)
        merge_base = merge_base_data["merge_base_commit"]["sha"]
        commit_data = self._get_compare_data(merge_base, commit_hash)

        # Extract components
        commits = self._extract_commits_from_response(commit_data)
        file_stats_str = self._build_file_stats_summary(commit_data)
        actual_diff = self._build_diff_content(commit_data)

        num_files = len(commit_data.get("files", []))
        logger.debug(f"GitHub API: Retrieved all data for {len(commits)} commits, {num_files} files")
        return merge_base, commits, file_stats_str, actual_diff

    def _extract_commits_from_response(self, commit_data: Dict[str, Any]) -> List[GitCommit]:
        """Extract GitCommit objects from GitHub API response."""
        commits = []
        for commit_item in commit_data["commits"]:
            commits.append(
                GitCommit(
                    hash=commit_item["sha"],
                    message=commit_item["commit"]["message"],
                    author=commit_item["commit"]["author"]["name"],
                    date=commit_item["commit"]["author"]["date"][:10],
                )
            )
        return commits

    def _build_file_stats_summary(self, commit_data: Dict[str, Any]) -> str:
        """Build file statistics summary from GitHub API response."""
        files = commit_data.get("files", [])
        files_changed = len(files)
        total_additions = sum(f.get("additions", 0) for f in files)
        total_deletions = sum(f.get("deletions", 0) for f in files)

        # Build summary line
        diff_stats = [f" {files_changed} files changed"]
        if total_additions > 0:
            diff_stats.append(f"{total_additions} insertions(+)")
        if total_deletions > 0:
            diff_stats.append(f"{total_deletions} deletions(-)")

        # Build individual file stats
        file_stats = []
        for file_info in files:
            filename = file_info.get("filename", "")
            additions = file_info.get("additions", 0)
            deletions = file_info.get("deletions", 0)
            changes = file_info.get("changes", 0)

            file_line = f" {filename}"
            if changes > 0:
                file_line += f" | {changes} "
                # Use proportional scaling instead of arbitrary limits
                file_line += self._build_change_indicator(additions, deletions)
            file_stats.append(file_line)

        # Combine file stats with summary
        result = "\n".join(file_stats)
        if diff_stats:
            result += "\n " + ", ".join(diff_stats)
        return result

    def _build_diff_content(self, commit_data: Dict[str, Any]) -> str:
        """Build unified diff content from GitHub API response."""
        diff_parts = []

        for file_info in commit_data.get("files", []):
            filename = file_info.get("filename", "")
            patch = file_info.get("patch", "")

            if patch:
                # Use more robust diff header construction
                diff_header = self._build_diff_header(filename, file_info)
                diff_parts.extend([diff_header, patch, ""])

        return "\n".join(diff_parts)

    def _build_diff_header(self, filename: str, file_info: Dict[str, Any]) -> str:
        """
        Build a proper git diff header for a file.

        Args:
            filename: The file name
            file_info: File information from GitHub API

        Returns:
            Formatted diff header
        """
        # Handle different file statuses (added, deleted, renamed, etc.)
        status = file_info.get("status", "modified")

        sha = file_info.get("sha", "unknown")[:7]

        if status == "added":
            return (
                f"diff --git a/{filename} b/{filename}\n"
                f"new file mode 100644\n"
                f"index 0000000..{sha}\n"
                f"--- /dev/null\n"
                f"+++ b/{filename}"
            )
        elif status == "deleted":
            return (
                f"diff --git a/{filename} b/{filename}\n"
                f"deleted file mode 100644\n"
                f"index {sha}..0000000\n"
                f"--- a/{filename}\n"
                f"+++ /dev/null"
            )
        elif status == "renamed":
            previous_filename = file_info.get("previous_filename", filename)
            changes = file_info.get("changes", 0)
            return (
                f"diff --git a/{previous_filename} b/{filename}\n"
                f"similarity index {changes}%\n"
                f"rename from {previous_filename}\n"
                f"rename to {filename}\n"
                f"index {sha}..{sha}\n"
                f"--- a/{previous_filename}\n"
                f"+++ b/{filename}"
            )
        else:
            # Standard modification
            return (
                f"diff --git a/{filename} b/{filename}\nindex {sha}..{sha} 100644\n--- a/{filename}\n+++ b/{filename}"
            )

    def _build_change_indicator(self, additions: int, deletions: int, max_width: int = 40) -> str:
        """
        Build a proportional visual indicator for file changes.

        Args:
            additions: Number of lines added
            deletions: Number of lines deleted
            max_width: Maximum width for the indicator

        Returns:
            String with + and - characters representing changes proportionally
        """
        if additions == 0 and deletions == 0:
            return ""

        total_changes = additions + deletions
        if total_changes <= max_width:
            # Small changes - show exactly
            return "+" * additions + "-" * deletions
        else:
            # Large changes - scale proportionally
            add_chars = int((additions / total_changes) * max_width)
            del_chars = max_width - add_chars  # Ensure we use the full width
            return "+" * add_chars + "-" * del_chars
