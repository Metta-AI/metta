"""PR splitting functionality."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anthropic import Anthropic  # type: ignore[import-not-found]

from .core import GitError, run_git
from .git import get_current_branch, get_remote_url
from .github import create_pr


@dataclass
class FileDiff:
    """Represents a diff for a single file"""

    filename: str
    additions: List[str]
    deletions: List[str]
    hunks: List[Dict[str, Any]]
    raw_diff: str


@dataclass
class SplitDecision:
    """Represents how to split the PR"""

    group1_files: List[str]
    group2_files: List[str]
    group1_description: str
    group2_description: str
    group1_title: str
    group2_title: str


class PRSplitter:
    """Split large pull requests into smaller, logically isolated ones."""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        github_token: Optional[str] = None,
        anthropic_client: Optional[Anthropic] = None,
        force_push: bool = True,
        independence: float = 0.5,
    ):
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic: Optional[Anthropic] = anthropic_client
        if self.anthropic is None and self.anthropic_api_key:
            self.anthropic = Anthropic(api_key=self.anthropic_api_key)

        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.base_branch: Optional[str] = None
        self.current_branch: Optional[str] = None
        self.force_push = force_push
        if not 0.0 <= independence <= 1.0:
            raise ValueError("Independence must be between 0.0 and 1.0")
        self.independence = independence

    def _ensure_anthropic(self) -> Anthropic:
        """Returns an Anthropic client, constructing it lazily if needed."""
        if self.anthropic:
            return self.anthropic

        if self.anthropic_api_key:
            self.anthropic = Anthropic(api_key=self.anthropic_api_key)
            return self.anthropic

        raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")

    def _build_split_prompt(self, file_summaries: List[Dict[str, Any]]) -> str:
        """Construct the AI prompt with independence guidance."""
        independence = self.independence
        balance_weight = 1.0 - independence

        independence_guidance = (
            "Prioritize balanced group sizes even if some files are loosely related."
            if independence <= 0.25
            else "Balance both logical separation and size; prefer independence when it doesn't create huge imbalance."
            if independence < 0.75
            else "Prioritize independence of concerns even if one group becomes much smaller."
        )

        prompt = f"""Analyze these file changes and suggest how to split them into two pull requests.

Files changed:
{json.dumps(file_summaries, indent=2)}

Independence preference: {independence:.2f}
- 0.00 means keep PR sizes balanced even if concerns overlap.
- 1.00 means maximize logical independence even if PR sizes differ greatly.
- Weight of balanced sizing: {balance_weight:.2f}
- Weight of independence: {independence:.2f}
- {independence_guidance}

Provide two groups that follow these guidelines:
1. Each PR should remain independently mergeable with minimal cross-dependencies.
2. Prefer grouping files by shared concern or feature area.
3. Keep PR sizes reasonably close unless independence weight is very high.

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
        return prompt

    def get_base_branch(self) -> str:
        """Determine the base branch (usually main or master)"""
        # Try to find the merge base with common default branches
        for branch in ["main", "master", "develop"]:
            try:
                run_git("rev-parse", f"origin/{branch}")
                return f"origin/{branch}"
            except GitError:
                continue
        raise GitError("Could not determine base branch")

    def get_diff(self, base: str, head: str) -> str:
        """Get the diff between two branches"""
        return run_git("diff", f"{base}...{head}")

    def parse_diff(self, diff_text: str) -> List[FileDiff]:
        """Parse git diff output into structured format"""
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

    def analyze_diff_with_ai(self, files: List[FileDiff]) -> SplitDecision:
        """Use Anthropic API to analyze how to split the diff"""

        anthropic_client = self._ensure_anthropic()

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

        prompt = self._build_split_prompt(file_summaries)

        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=1000, messages=[{"role": "user", "content": prompt}]
        )

        # Parse the AI response
        content = response.content[0].text

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

    def create_patch_file(self, files: List[FileDiff], selected_files: List[str]) -> str:
        """Create a patch file containing only the selected files"""
        patch_content = []

        for f in files:
            if f.filename in selected_files:
                patch_content.append(f.raw_diff)

        return "\n".join(patch_content)

    def apply_patch_to_new_branch(self, patch_content: str, branch_name: str) -> None:
        """Create a new branch and apply the patch"""
        if self.base_branch is None:
            raise GitError("Base branch not set. Run split() before applying patches.")

        base_branch = self.base_branch

        # Create and checkout new branch from base
        run_git("checkout", "-B", branch_name, base_branch)

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

    def verify_split(self, original_diff: str, diff1: str, diff2: str) -> bool:
        """Verify that the split diffs contain all changes from the original"""
        # Simple verification: check that all lines are accounted for
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

    def push_branch(self, branch_name: str) -> None:
        """Push a branch, retrying with --force-with-lease if the remote already exists."""
        try:
            run_git("push", "origin", branch_name)
        except GitError as exc:
            message = str(exc).lower()
            rerun_indicators = ("non-fast-forward", "fetch first", "updates were rejected because", "rejected")
            if not any(indicator in message for indicator in rerun_indicators):
                raise

            if not self.force_push:
                raise

            print(f"Remote branch {branch_name} already exists. Updating with --force-with-lease.")
            run_git("push", "--force-with-lease", "origin", branch_name)

    def get_repo_from_remote(self) -> Optional[str]:
        """Extract owner/repo from git remote URL"""
        try:
            remote_url = get_remote_url()
            if not remote_url:
                return None

            # Extract owner/repo from URL
            match = re.search(r"github\.com[:/](.+/.+?)(?:\.git)?$", remote_url)
            if match:
                return match.group(1)

            return None
        except GitError:
            return None

    def create_github_pr(self, branch: str, title: str, body: str) -> Optional[str]:
        """Create a GitHub PR using the API"""
        if self.base_branch is None:
            raise GitError("Base branch not set. Run split() before creating PRs.")

        if not self.github_token:
            print(f"Skipping PR creation for {branch} (no GitHub token provided)")
            return None

        # Get repo info
        repo = self.get_repo_from_remote()
        if not repo:
            print("Could not determine GitHub repository from remote")
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
        """Main method to split the current PR"""
        print("🔄 Starting PR split process...")

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

        # Verify all files are accounted for
        all_files = set(f.filename for f in files)
        assigned_files = set(split_decision.group1_files + split_decision.group2_files)

        if all_files != assigned_files:
            print("\n⚠️  Warning: File mismatch!")
            print(f"  Unassigned: {all_files - assigned_files}")
            print(f"  Unknown: {assigned_files - all_files}")

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

        # Create branches
        branch1_name = f"{current_branch}-part1"
        branch2_name = f"{current_branch}-part2"

        print(f"\n🌿 Creating branch: {branch1_name}")
        self.apply_patch_to_new_branch(patch1, branch1_name)
        run_git("commit", "-m", split_decision.group1_title)

        print(f"🌿 Creating branch: {branch2_name}")
        self.apply_patch_to_new_branch(patch2, branch2_name)
        run_git("commit", "-m", split_decision.group2_title)

        # Push branches
        print("\n📤 Pushing branches...")
        self.push_branch(branch1_name)
        self.push_branch(branch2_name)

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
    anthropic_api_key: Optional[str] = None,
    github_token: Optional[str] = None,
    *,
    force_push: bool = True,
    independence: float = 0.5,
) -> None:
    """
    Split the current branch into two smaller PRs.

    Args:
        anthropic_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        github_token: GitHub token (defaults to GITHUB_TOKEN env var)
        force_push: Retry push with --force-with-lease when remote branches exist (default True)
        independence: Weight between balanced sizing (0.0) and logical separation (1.0)
    """
    splitter = PRSplitter(
        anthropic_api_key,
        github_token,
        force_push=force_push,
        independence=independence,
    )
    splitter.split()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Split large PRs into smaller, logically isolated ones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split current branch using environment variables
  python -m gitta.split

  # Split with explicit API key
  python -m gitta.split --anthropic-key YOUR_KEY

  # Favor balanced PR sizes
  python -m gitta.split --independence 0.1

  # Also create GitHub PRs
  python -m gitta.split --github-token YOUR_TOKEN

Environment variables:
  ANTHROPIC_API_KEY - Anthropic API key for AI analysis
  GITHUB_TOKEN      - GitHub token for creating PRs (optional)
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
        "--independence",
        type=float,
        default=0.5,
        help="Balance between equal-sized splits (0.0) and logical separation (1.0). Default: 0.5",
    )

    parser.add_argument(
        "--no-force-push",
        action="store_true",
        help="Skip the force-with-lease retry when remote branches already exist",
    )

    args = parser.parse_args()

    # Validate API key
    if not args.anthropic_key:
        print("❌ Error: Anthropic API key not provided!")
        print("   Set ANTHROPIC_API_KEY environment variable or use --anthropic-key")
        sys.exit(1)

    if not args.github_token:
        print("⚠️  Warning: GitHub token not provided, will skip PR creation")
        print("   Set GITHUB_TOKEN environment variable or use --github-token to enable")

    if not 0.0 <= args.independence <= 1.0:
        print("❌ Error: --independence must be between 0.0 and 1.0")
        sys.exit(1)

    try:
        split_pr(
            anthropic_api_key=args.anthropic_key,
            github_token=args.github_token,
            force_push=not args.no_force_push,
            independence=args.independence,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
