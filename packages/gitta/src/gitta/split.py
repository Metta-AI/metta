"""PR splitting functionality."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import TextBlock

from .core import GitError, run_git, run_git_cmd
from .git import get_current_branch, get_remote_url
from .github import create_pr


@dataclass
class FileDiff:
    filename: str
    additions: list[str]
    deletions: list[str]
    hunks: list[dict[str, Any]]
    raw_diff: str


@dataclass
class SplitDecision:
    group1_files: list[str]
    group2_files: list[str]
    group1_description: str
    group2_description: str
    group1_title: str
    group2_title: str


DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_COMMIT_TIMEOUT = 300.0


class PRSplitter:
    """Split large pull requests into smaller, logically isolated ones."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        github_token: str | None = None,
        model: str | None = None,
        skip_hooks: bool | None = None,
        commit_timeout: float | None = None,
    ):
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._anthropic: Anthropic | None = None
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.model = model or os.environ.get("GITTA_SPLIT_MODEL") or DEFAULT_MODEL

        if skip_hooks is None:
            skip_hooks_env = os.environ.get("GITTA_SKIP_HOOKS", "").lower()
            self.skip_hooks = skip_hooks_env in {"1", "true", "yes", "on"}
        else:
            self.skip_hooks = skip_hooks

        if commit_timeout is not None:
            self.commit_timeout = commit_timeout
        else:
            timeout_env = os.environ.get("GITTA_COMMIT_TIMEOUT")
            if timeout_env:
                try:
                    self.commit_timeout = float(timeout_env)
                except ValueError as exc:
                    raise ValueError("GITTA_COMMIT_TIMEOUT must be a number") from exc
            else:
                self.commit_timeout = DEFAULT_COMMIT_TIMEOUT

        self.base_branch: str | None = None
        self.current_branch: str | None = None

    def _get_anthropic_client(self) -> Anthropic:
        api_key = self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")

        if self._anthropic is None or api_key != self.anthropic_api_key:
            self.anthropic_api_key = api_key
            self._anthropic = Anthropic(api_key=api_key)

        return self._anthropic

    def get_base_branch(self) -> str:
        # Try to find the merge base with common default branches
        for branch in ["main", "master", "develop"]:
            try:
                run_git("rev-parse", f"origin/{branch}")
                return f"origin/{branch}"
            except GitError:
                continue
        raise GitError("Could not determine base branch")

    def get_diff(self, base: str, head: str) -> str:
        return run_git("diff", f"{base}...{head}")

    def parse_diff(self, diff_text: str) -> list[FileDiff]:
        files = []
        current_file = None

        lines = diff_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.startswith("diff --git"):
                # Save previous file if exists
                if current_file:
                    files.append(current_file)

                # Extract filename
                match = re.search(r"diff --git a/(.*) b/(.*)", line)
                if match:
                    filename = match.group(2)

                    # Collect the full diff for this file
                    file_start = i
                    i += 1
                    while i < len(lines) and not lines[i].startswith("diff --git"):
                        i += 1

                    raw_diff = "\n".join(lines[file_start:i])

                    # Parse hunks
                    hunks = []
                    additions = []
                    deletions = []

                    for diff_line in raw_diff.split("\n"):
                        if diff_line.startswith("@@"):
                            hunk_match = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)", diff_line)
                            if hunk_match:
                                hunks.append(
                                    {
                                        "old_start": int(hunk_match.group(1)),
                                        "old_lines": int(hunk_match.group(2) or 1),
                                        "new_start": int(hunk_match.group(3)),
                                        "new_lines": int(hunk_match.group(4) or 1),
                                        "header": diff_line,
                                    }
                                )
                        elif diff_line.startswith("+") and not diff_line.startswith("+++"):
                            additions.append(diff_line)
                        elif diff_line.startswith("-") and not diff_line.startswith("---"):
                            deletions.append(diff_line)

                    current_file = FileDiff(
                        filename=filename, additions=additions, deletions=deletions, hunks=hunks, raw_diff=raw_diff
                    )
                    i -= 1  # Back up one since we've already advanced
            i += 1

        # Don't forget the last file
        if current_file:
            files.append(current_file)

        return files

    def analyze_diff_with_ai(self, files: list[FileDiff]) -> SplitDecision:
        # Prepare a summary for the AI
        file_summaries = []
        for f in files:
            file_summaries.append(
                {
                    "filename": f.filename,
                    "additions": len(f.additions),
                    "deletions": len(f.deletions),
                    "hunks": len(f.hunks),
                    # Include first few lines of changes for context
                    "sample_changes": "\n".join(f.additions[:5] + f.deletions[:5]),
                }
            )

        prompt = f"""Analyze these file changes and suggest how to split them into two logically isolated pull requests.

Files changed:
{json.dumps(file_summaries, indent=2)}

Consider:
1. Logical grouping (e.g., feature vs tests, frontend vs backend)
2. Dependencies between files
3. Roughly equal size distribution
4. Each PR should be independently mergeable

Return a JSON response with this exact structure:
{{
    "group1_files": ["file1.py", "file2.py"],
    "group2_files": ["file3.js", "file4.js"],
    "group1_description": "Brief description of what group 1 does",
    "group2_description": "Brief description of what group 2 does",
    "group1_title": "Short PR title for group 1",
    "group2_title": "Short PR title for group 2"
}}
"""

        client = self._get_anthropic_client()

        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the AI response
        text_block = next((block for block in response.content if isinstance(block, TextBlock)), None)
        if text_block is None:
            raise ValueError("Anthropic response did not include a text block")
        content = cast(str, text_block.text)

        # Try to extract JSON from the response
        try:
            # Look for JSON in code blocks first
            json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            decision_data = json.loads(json_str)

            # Validate all required fields are present
            required_fields = [
                "group1_files",
                "group2_files",
                "group1_description",
                "group2_description",
                "group1_title",
                "group2_title",
            ]
            missing_fields = [field for field in required_fields if field not in decision_data]
            if missing_fields:
                raise ValueError(f"AI response missing required fields: {', '.join(missing_fields)}")

            # Validate that file lists are actually lists
            if not isinstance(decision_data["group1_files"], list):
                raise ValueError("group1_files must be a list")
            if not isinstance(decision_data["group2_files"], list):
                raise ValueError("group2_files must be a list")

            return SplitDecision(
                group1_files=decision_data["group1_files"],
                group2_files=decision_data["group2_files"],
                group1_description=decision_data["group1_description"],
                group2_description=decision_data["group2_description"],
                group1_title=decision_data["group1_title"],
                group2_title=decision_data["group2_title"],
            )

        except Exception as e:
            print(f"Failed to parse AI response: {e}")
            print(f"Response content: {content}")
            raise

    def create_patch_file(self, files: list[FileDiff], selected_files: list[str]) -> str:
        patch_content = []

        for f in files:
            if f.filename in selected_files:
                patch_content.append(f.raw_diff)

        if not patch_content:
            return ""

        patch = "\n".join(patch_content)
        return patch if patch.endswith("\n") else f"{patch}\n"

    def apply_patch_to_new_branch(self, patch_content: str, branch_name: str) -> None:
        if self.base_branch is None:
            raise ValueError("Base branch is not set")

        # Create and checkout new branch from base
        run_git("checkout", "-b", branch_name, self.base_branch)

        # Apply the patch
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch_content)
            patch_file = f.name

        try:
            run_git("apply", patch_file)

            # Stage all changes
            run_git("add", "-A")

        finally:
            os.unlink(patch_file)

    def commit_changes(self, message: str) -> None:
        commit_args = ["commit"]
        if self.skip_hooks:
            commit_args.append("--no-verify")
        commit_args.extend(["-m", message])
        run_git_cmd(commit_args, timeout=self.commit_timeout)

    def ensure_clean_worktree(self) -> None:
        status = run_git("status", "--porcelain")
        if status.strip():
            raise GitError(
                "Working tree has uncommitted changes. Please commit or stash them before running the PR splitter."
            )

    def verify_split(self, original_diff: str, diff1: str, diff2: str) -> bool:
        # Check that all lines are accounted for
        original_lines = set(
            line
            for line in original_diff.split("\n")
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )

        split_lines = set(
            line
            for line in (diff1 + "\n" + diff2).split("\n")
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )

        missing = original_lines - split_lines
        extra = split_lines - original_lines

        if missing:
            print("Warning: Missing lines in split:")
            for line in list(missing)[:10]:  # Show first 10
                print(f"  {line}")

        if extra:
            print("Warning: Extra lines in split:")
            for line in list(extra)[:10]:  # Show first 10
                print(f"  {line}")

        return len(missing) == 0 and len(extra) == 0

    def get_repo_from_remote(self) -> str | None:
        try:
            remote_url = get_remote_url()
            if not remote_url:
                return None

            # Extract owner/repo from URL
            match = re.search(r"github\.com[:/](.+/.+?)(?:\.git)?$", remote_url)
            if match:
                return match.group(1)

            return None
        except Exception:
            return None

    def branch_exists(self, branch_name: str) -> bool:
        try:
            # Check local branch
            run_git("rev-parse", "--verify", branch_name)
            return True
        except GitError:
            pass

        try:
            # Check remote branch
            run_git("rev-parse", "--verify", f"origin/{branch_name}")
            return True
        except GitError:
            pass

        return False

    def generate_unique_branch_name(self, base_name: str) -> str:
        if not self.branch_exists(base_name):
            return base_name

        # Branch exists, append timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{base_name}-{timestamp}"
        return new_name

    def create_github_pr(self, branch: str, title: str, body: str) -> str | None:
        if not self.github_token:
            print(f"Skipping PR creation for {branch} (no GitHub token provided)")
            return None

        # Get repo info
        repo = self.get_repo_from_remote()
        if not repo:
            print("Could not determine GitHub repository from remote")
            return None

        if self.base_branch is None:
            print("Base branch not set; skipping PR creation")
            return None

        # Get base branch name without origin/
        base_branch_name = self.base_branch.replace("origin/", "")

        try:
            pr_data = create_pr(
                repo=repo, title=title, body=body, head=branch, base=base_branch_name, token=self.github_token
            )
            print(f"Created PR: {pr_data['html_url']}")
            return pr_data["html_url"]
        except Exception as e:
            print(f"Failed to create PR: {e}")
            return None

    def split(self) -> None:
        print("🔄 Starting PR split process...")
        self.ensure_clean_worktree()

        # Get branch information
        current_branch = get_current_branch()
        base_branch = self.get_base_branch()

        self.current_branch = current_branch
        self.base_branch = base_branch

        print(f"📍 Current branch: {current_branch}")
        print(f"📍 Base branch: {base_branch}")

        # Get the full diff
        print("📥 Getting diff...")
        full_diff = self.get_diff(base_branch, current_branch)

        if not full_diff:
            print("❌ No changes detected!")
            return

        # Parse the diff
        files = self.parse_diff(full_diff)
        print(f"📊 Found {len(files)} changed files")

        if len(files) < 2:
            print("❌ Need at least 2 files to split!")
            return

        # Analyze with AI
        print("🤖 Analyzing with AI to determine split strategy...")
        split_decision = self.analyze_diff_with_ai(files)

        print("\n📋 Split Decision:")
        print(f"Group 1 ({len(split_decision.group1_files)} files): {split_decision.group1_title}")
        print(f"  Files: {', '.join(split_decision.group1_files)}")
        print(f"  Description: {split_decision.group1_description}")
        print(f"\nGroup 2 ({len(split_decision.group2_files)} files): {split_decision.group2_title}")
        print(f"  Files: {', '.join(split_decision.group2_files)}")
        print(f"  Description: {split_decision.group2_description}")

        # Validate that both groups have at least one file
        if not split_decision.group1_files:
            raise GitError("Cannot split: Group 1 is empty. Both groups must contain at least one file.")
        if not split_decision.group2_files:
            raise GitError("Cannot split: Group 2 is empty. Both groups must contain at least one file.")

        # Verify all files are accounted for
        all_files = set(f.filename for f in files)
        assigned_files = set(split_decision.group1_files + split_decision.group2_files)

        if all_files != assigned_files:
            unassigned = all_files - assigned_files
            unknown = assigned_files - all_files

            error_parts = []
            if unassigned:
                error_parts.append(f"Unassigned files: {', '.join(sorted(unassigned))}")
            if unknown:
                error_parts.append(f"Unknown files (not in diff): {', '.join(sorted(unknown))}")

            raise GitError(
                f"AI split decision has file mismatch. {' | '.join(error_parts)}. "
                "Please retry the split operation or manually adjust the split."
            )

        # Create patches
        print("\n✂️  Creating patches...")
        patch1 = self.create_patch_file(files, split_decision.group1_files)
        patch2 = self.create_patch_file(files, split_decision.group2_files)

        # Verify the split
        print("✅ Verifying split integrity...")
        if self.verify_split(full_diff, patch1, patch2):
            print("  Verification passed!")
        else:
            print("  ⚠️  Verification warnings detected")

        # Create branches with unique names
        branch1_name = self.generate_unique_branch_name(f"{current_branch}-part1")
        branch2_name = self.generate_unique_branch_name(f"{current_branch}-part2")

        print(f"\n🌿 Creating branch: {branch1_name}")
        self.apply_patch_to_new_branch(patch1, branch1_name)
        self.commit_changes(split_decision.group1_title)

        print(f"🌿 Creating branch: {branch2_name}")
        self.apply_patch_to_new_branch(patch2, branch2_name)
        self.commit_changes(split_decision.group2_title)

        # Push branches
        print("\n📤 Pushing branches...")
        run_git("push", "origin", branch1_name)
        run_git("push", "origin", branch2_name)

        # Create PRs
        print("\n🔧 Creating pull requests...")
        pr1_body = f"{split_decision.group1_description}\n\nThis PR is part 1 of a split from `{self.current_branch}`"
        pr2_body = f"{split_decision.group2_description}\n\nThis PR is part 2 of a split from `{self.current_branch}`"

        self.create_github_pr(branch1_name, split_decision.group1_title, pr1_body)
        self.create_github_pr(branch2_name, split_decision.group2_title, pr2_body)

        # Return to original branch
        run_git("checkout", current_branch)

        print("\n✨ PR split complete!")


def split_pr(
    anthropic_api_key: str | None = None,
    github_token: str | None = None,
    model: str | None = None,
    skip_hooks: bool | None = None,
    commit_timeout: float | None = None,
) -> None:
    """Split the current branch into two smaller PRs using AI analysis."""
    splitter = PRSplitter(
        anthropic_api_key=anthropic_api_key,
        github_token=github_token,
        model=model,
        skip_hooks=skip_hooks,
        commit_timeout=commit_timeout,
    )
    splitter.split()


def main():
    parser = argparse.ArgumentParser(
        description="Split large PRs into smaller, logically isolated ones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Split current branch using environment variables
  python -m gitta.split

  # Split with explicit API key
  python -m gitta.split --anthropic-key YOUR_KEY

  # Also create GitHub PRs
  python -m gitta.split --github-token YOUR_TOKEN

  # Override the Anthropic model
  python -m gitta.split --model {DEFAULT_MODEL}

Environment variables:
  ANTHROPIC_API_KEY - Anthropic API key for AI analysis
  GITHUB_TOKEN      - GitHub token for creating PRs (optional)
  GITTA_SPLIT_MODEL - Anthropic model name (optional, defaults to {DEFAULT_MODEL})
  GITTA_SKIP_HOOKS  - Set to "1" to skip git hooks when committing split branches
  GITTA_COMMIT_TIMEOUT - Commit timeout in seconds (defaults to {DEFAULT_COMMIT_TIMEOUT})
        """,
    )

    parser.add_argument(
        "--anthropic-key",
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)",
        default=os.environ.get("ANTHROPIC_API_KEY"),
    )

    parser.add_argument(
        "--github-token",
        help="GitHub token for creating PRs (defaults to GITHUB_TOKEN env var)",
        default=os.environ.get("GITHUB_TOKEN"),
    )

    parser.add_argument(
        "--model",
        help=f"Anthropic model to use (defaults to GITTA_SPLIT_MODEL env var or {DEFAULT_MODEL})",
        default=os.environ.get("GITTA_SPLIT_MODEL"),
    )
    parser.add_argument(
        "--skip-hooks",
        dest="skip_hooks",
        action="store_true",
        help="Skip git hooks (--no-verify) when committing split branches",
    )
    parser.add_argument(
        "--no-skip-hooks",
        dest="skip_hooks",
        action="store_false",
        help="Run git hooks when committing split branches",
    )
    parser.add_argument(
        "--commit-timeout",
        type=float,
        default=None,
        help=(
            f"Timeout in seconds for git commit (defaults to GITTA_COMMIT_TIMEOUT env var or {DEFAULT_COMMIT_TIMEOUT})"
        ),
    )
    parser.set_defaults(skip_hooks=None)

    args = parser.parse_args()

    # Validate API key
    if not args.anthropic_key:
        print("❌ Error: Anthropic API key not provided!")
        print("   Set ANTHROPIC_API_KEY environment variable or use --anthropic-key")
        sys.exit(1)

    if not args.github_token:
        print("⚠️  Warning: GitHub token not provided, will skip PR creation")
        print("   Set GITHUB_TOKEN environment variable or use --github-token to enable")

    try:
        split_pr(
            anthropic_api_key=args.anthropic_key,
            github_token=args.github_token,
            model=args.model,
            skip_hooks=args.skip_hooks,
            commit_timeout=args.commit_timeout,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
