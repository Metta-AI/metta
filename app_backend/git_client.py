# NOTE: This file duplicates functionality with common/src/metta/common/util/git.py
# Both files implement similar git operations but with different interfaces.
# TODO: Extract common git functionality into a shared library to avoid duplication.

import logging
import subprocess
from typing import List

from app_backend.github_client import GitCommit

logger = logging.getLogger("git_client")


class GitError(Exception):
    """Custom exception for git-related errors."""


def run_git(*args: str) -> str:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"Git command failed ({e.returncode}): {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise GitError("Git is not installed!") from e


class GitClient:
    """Client for git operations, mimicking the existing metta git utilities."""

    DEFAULT_BRANCH = "main"

    def get_current_commit(self) -> str:
        """Get the current git commit hash."""
        return run_git("rev-parse", "HEAD")

    def get_current_branch(self) -> str:
        """Get the current git branch name."""
        try:
            return run_git("symbolic-ref", "--short", "HEAD")
        except GitError as e:
            if "not a git repository" in str(e):
                raise ValueError("Not in a git repository") from e
            elif "HEAD is not a symbolic ref" in str(e):
                return self.get_current_commit()
            raise

    def commit_exists(self, commit_hash: str) -> bool:
        """Check if commit exists locally."""
        try:
            run_git("cat-file", "-e", commit_hash)
            return True
        except GitError:
            return False

    def get_merge_base(self, commit_hash: str, base_branch: str = DEFAULT_BRANCH) -> str:
        """Get merge base between commit and base branch."""
        try:
            return run_git("merge-base", base_branch, commit_hash)
        except GitError:
            # Fallback to base branch
            logger.warning(f"Could not find merge base for {commit_hash}, using {base_branch}")
            return base_branch

    def _parse_commit_line(self, line: str) -> GitCommit:
        """Parse a single commit line in format 'hash|message|author|date'."""
        parts = line.split("|", 3)
        if len(parts) == 4:
            hash_val, message, author, date = parts
            return GitCommit(hash_val, message, author, date)
        raise ValueError(f"Invalid commit line format: {line}")

    def get_commit_range(self, commit_hash: str, base_branch: str = DEFAULT_BRANCH) -> List[GitCommit]:
        """Get commits from merge base to specified commit."""
        # Handle edge case where commit_hash equals base_branch
        if commit_hash == base_branch:
            # Return the single commit itself
            try:
                result = run_git("log", "--pretty=format:%H|%s|%an|%ad", "--date=short", "-1", commit_hash)
                lines = result.strip().split("\n")
                if lines and lines[0]:
                    return [self._parse_commit_line(lines[0])]
            except GitError:
                pass
            return []

        merge_base = self.get_merge_base(commit_hash, base_branch)
        commit_range = f"{merge_base}..{commit_hash}"

        try:
            result = run_git("log", "--pretty=format:%H|%s|%an|%ad", "--date=short", "--reverse", commit_range)

            if not result.strip():
                # Single commit case
                result = run_git("log", "--pretty=format:%H|%s|%an|%ad", "--date=short", "-n", "1", commit_hash)

            commits = []
            for line in result.strip().split("\n"):
                if line:
                    try:
                        commits.append(self._parse_commit_line(line))
                    except ValueError:
                        # Skip invalid lines
                        continue

            return commits

        except GitError as e:
            raise ValueError(f"Could not retrieve commit history: {str(e)}") from e

    def get_range_diff_stats(self, commit_hash: str, base_branch: str = DEFAULT_BRANCH) -> str:
        """Get the diff file statistics for the entire range from merge base to commit."""
        merge_base = self.get_merge_base(commit_hash, base_branch)

        try:
            # Get the diff with file stats
            result = run_git("diff", "--stat", f"{merge_base}..{commit_hash}")
            return result
        except GitError as e:
            logger.warning(f"Could not get range diff stats: {str(e)}")
            return ""

    def get_range_diff(self, commit_hash: str, base_branch: str = DEFAULT_BRANCH) -> str:
        """Get the actual diff content for the entire range from merge base to commit."""
        merge_base = self.get_merge_base(commit_hash, base_branch)

        try:
            # Get the actual diff with + and - lines
            result = run_git("diff", f"{merge_base}..{commit_hash}")
            return result
        except GitError as e:
            logger.warning(f"Could not get range diff: {str(e)}")
            return ""
