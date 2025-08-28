"""
Git layer for codebot with AI commit tracking and rollback functionality.

This module provides high-level git operations specifically designed for
tracking AI-generated changes and enabling rollback to human commits.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import gitta
from gitta import GitError

from .models import FileChange

logger = logging.getLogger(__name__)

# Constants for AI commit identification
AI_COMMIT_PREFIX = "[CODEBOT]"
AI_AUTHOR_NAME = "Codebot AI"
AI_AUTHOR_EMAIL = "codebot@ai.generated"


# Helper functions to bridge our API with gitta
def _add_all(repo_root: Path) -> None:
    """Add all changes to the staging area."""
    gitta.run_git_in_dir(repo_root, "add", ".")


def _commit_with_author(repo_root: Path, message: str, author_name: str, author_email: str) -> str:
    """Create a commit with custom author. Returns commit hash."""
    gitta.run_git_in_dir(repo_root, "commit", "-m", message, "--author", f"{author_name} <{author_email}>")
    return gitta.run_git_in_dir(repo_root, "rev-parse", "HEAD")


def _get_last_commits(repo_root: Path, count: int = 20) -> List[dict]:
    """Get the last N commits with hash and message."""
    try:
        # Format: hash|subject
        output = gitta.run_git_in_dir(repo_root, "log", f"--max-count={count}", "--format=%H|%s")
        commits = []

        for line in output.split("\n"):
            if "|" in line:
                hash_part, subject = line.split("|", 1)
                commits.append({"hash": hash_part.strip(), "subject": subject.strip()})

        return commits
    except GitError:
        return []


def _reset_soft(repo_root: Path, commit_hash: str) -> None:
    """Reset to a commit using --soft (keeps changes in staging area)."""
    gitta.run_git_in_dir(repo_root, "reset", "--soft", commit_hash)


def _has_uncommitted_changes(repo_root: Path) -> bool:
    """Check if there are uncommitted changes in the working directory."""
    try:
        # gitta.has_unstaged_changes returns (bool, output) tuple
        has_changes, _ = gitta.has_unstaged_changes(allow_untracked=False)
        return has_changes
    except GitError:
        return False


class GitLayerError(Exception):
    """Custom exception for GitLayer operations."""

    pass


class GitLayer:
    """
    High-level git operations for codebot with AI commit tracking.

    Provides functionality to:
    - Commit AI-generated changes with special formatting
    - Track the distinction between human and AI commits
    - Rollback to the last human commit
    - Detect AI vs human commits
    """

    def __init__(self, working_dir: Optional[Path] = None):
        """Initialize GitLayer with optional working directory."""
        self.working_dir = working_dir or Path.cwd()
        self.repo_root = gitta.find_root(self.working_dir)

        if not self.repo_root:
            raise GitLayerError(f"No git repository found in {self.working_dir}")

    def commit_ai_changes(self, command: str, changes: List[FileChange], description: str = "") -> str:
        """
        Commit AI-generated changes with special formatting.

        Args:
            command: The codebot command that generated these changes (e.g., "test", "refactor")
            changes: List of FileChange objects that were applied
            description: Optional description of what the AI did

        Returns:
            Commit hash of the created commit

        Raises:
            GitLayerError: If there are no changes to commit or git operations fail
        """
        try:
            # Check if there are actually changes to commit
            if not _has_uncommitted_changes(self.repo_root):
                raise GitLayerError("No uncommitted changes to commit")

            # Stage all changes
            _add_all(self.repo_root)

            # Format commit message
            file_list = [change.filepath for change in changes]
            summary = self._generate_change_summary(changes)

            commit_message = self._format_ai_commit_message(
                command=command, summary=summary, description=description, files=file_list
            )

            # Create commit with AI author
            commit_hash = _commit_with_author(self.repo_root, commit_message, AI_AUTHOR_NAME, AI_AUTHOR_EMAIL)

            logger.info(f"Created AI commit {commit_hash[:8]} for '{command}' command")
            return commit_hash

        except GitError as e:
            raise GitLayerError(f"Failed to commit AI changes: {e}") from e

    def find_last_human_commit(self) -> Optional[str]:
        """
        Find the most recent commit that was not made by AI.

        Returns:
            Commit hash of the last human commit, or None if no human commits found
        """
        commits = _get_last_commits(self.repo_root, count=50)  # Look back 50 commits

        for commit_info in commits:
            if not self.is_ai_commit(commit_info["hash"]):
                logger.debug(f"Found last human commit: {commit_info['hash'][:8]} - {commit_info['subject']}")
                return commit_info["hash"]

        logger.warning("No human commits found in recent history")
        return None

    def reset_to_human(self) -> Optional[str]:
        """
        Reset (soft) to the last human commit, undoing all AI commits.

        Returns:
            Hash of the commit that was reset to, or None if no human commit found

        Raises:
            GitLayerError: If reset operation fails
        """
        last_human_commit = self.find_last_human_commit()

        if not last_human_commit:
            raise GitLayerError("No human commit found to reset to")

        try:
            _reset_soft(self.repo_root, last_human_commit)
            logger.info(f"Reset to last human commit: {last_human_commit[:8]}")
            return last_human_commit
        except GitError as e:
            raise GitLayerError(f"Failed to reset to human commit: {e}") from e

    def is_ai_commit(self, commit_hash: str) -> bool:
        """
        Check if a commit was made by AI.

        Args:
            commit_hash: Git commit hash to check

        Returns:
            True if commit was made by AI, False otherwise
        """
        commit_message = gitta.get_commit_message(commit_hash)

        # Check for AI commit prefix in message
        if commit_message.startswith(AI_COMMIT_PREFIX):
            return True

        # Could also check author, but message prefix is more reliable
        return False

    def get_ai_commits_since_human(self) -> List[dict]:
        """
        Get all AI commits since the last human commit.

        Returns:
            List of commit info dicts with 'hash' and 'subject' keys
        """
        commits = _get_last_commits(self.repo_root, count=50)
        ai_commits = []

        for commit_info in commits:
            if self.is_ai_commit(commit_info["hash"]):
                ai_commits.append(commit_info)
            else:
                # Stop at first human commit
                break

        return ai_commits

    def _format_ai_commit_message(self, command: str, summary: str, description: str, files: List[str]) -> str:
        """Format a commit message for AI-generated changes."""
        # Main title with command
        title = f"{AI_COMMIT_PREFIX} {command}: {summary}"

        # Body with details
        body_parts = [
            f"AI-generated changes via codebot {command}",
            f"Timestamp: {datetime.now().isoformat()}",
        ]

        if description:
            body_parts.append(f"Description: {description}")

        if files:
            # Limit file list to avoid overly long commit messages
            if len(files) <= 5:
                body_parts.append(f"Files: {', '.join(files)}")
            else:
                body_parts.append(f"Files: {', '.join(files[:5])} and {len(files) - 5} more")

        return title + "\n\n" + "\n".join(body_parts)

    def _generate_change_summary(self, changes: List[FileChange]) -> str:
        """Generate a concise summary of what changes were made."""
        if not changes:
            return "No changes"

        file_count = len(changes)

        # Categorize changes
        new_files = [c for c in changes if c.operation == "write" and not Path(c.filepath).exists()]
        modified_files = [c for c in changes if c.operation == "write" and Path(c.filepath).exists()]
        deleted_files = [c for c in changes if c.operation == "delete"]

        parts = []
        if new_files:
            parts.append(f"created {len(new_files)} file{'s' if len(new_files) > 1 else ''}")
        if modified_files:
            parts.append(f"modified {len(modified_files)} file{'s' if len(modified_files) > 1 else ''}")
        if deleted_files:
            parts.append(f"deleted {len(deleted_files)} file{'s' if len(deleted_files) > 1 else ''}")

        if parts:
            return ", ".join(parts)
        else:
            return f"updated {file_count} file{'s' if file_count > 1 else ''}"
