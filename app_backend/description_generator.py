import os
import subprocess
from pathlib import Path
from typing import Optional

try:
    import anthropic  # type: ignore[import-untyped]

    _anthropic_available = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    _anthropic_available = False


class DescriptionGenerator:
    """Service for generating training run descriptions using Claude."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the description generator.

        Args:
            api_key: Anthropic API key. If not provided, will try to get from environment.

        Raises:
            ImportError: If anthropic package is not available
            ValueError: If API key is not provided and not in environment
        """
        if not _anthropic_available or anthropic is None:
            raise ImportError("anthropic package is not installed. Install with 'pip install anthropic'")

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"  # Using a capable model for better code analysis

    def get_git_diff_from_main(self, git_hash: str, repo_path: Optional[Path] = None) -> str:
        """Get the git diff between the specified commit and its branch point from main.

        Args:
            git_hash: The git commit hash to diff against its branch point from main
            repo_path: Path to the git repository. If None, uses the current working directory.

        Returns:
            The git diff as a string
        """
        if repo_path is None:
            # Default to the repo root (assuming we're running from within the repo)
            repo_path = Path(__file__).parent.parent

        try:
            # First, find the merge base (common ancestor) between main and the commit
            merge_base_result = subprocess.run(
                ["git", "merge-base", "main", git_hash], cwd=repo_path, capture_output=True, text=True, check=True
            )
            merge_base = merge_base_result.stdout.strip()

            # Now diff between the merge base and the training run commit
            result = subprocess.run(
                ["git", "diff", merge_base, git_hash], cwd=repo_path, capture_output=True, text=True, check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            # If merge-base fails, try the original approach as fallback
            try:
                result = subprocess.run(
                    ["git", "diff", f"main...{git_hash}"], cwd=repo_path, capture_output=True, text=True, check=True
                )
                return result.stdout
            except subprocess.CalledProcessError:
                # Last resort: show changes in the commit itself
                result = subprocess.run(
                    ["git", "show", "--format=", git_hash], cwd=repo_path, capture_output=True, text=True, check=True
                )
                return result.stdout

    def generate_description(self, git_hash: str) -> str:
        """Generate a training run description based on the git changes.

        Args:
            git_hash: The git commit hash for this training run

        Returns:
            A generated description of the training run
        """
        try:
            git_diff = self.get_git_diff_from_main(git_hash)
        except Exception as e:
            return f"Unable to generate description: Could not retrieve git diff for {git_hash[:8]}: {str(e)}"

        if not git_diff.strip():
            return "No changes detected compared to main branch"

        # Truncate very long diffs to avoid token limits
        max_diff_length = 8000  # Conservative limit for Claude
        if len(git_diff) > max_diff_length:
            git_diff = git_diff[:max_diff_length] + "\n... (diff truncated due to length)"

        prompt = (
            f"Please analyze this git diff and generate a concise, informative description "
            f"for a machine learning training run. The description should be 1-3 sentences "
            f"and focus on the key changes that would affect the training behavior.\n\n"
            f"Focus on:\n"
            f"- Changes to model architecture, hyperparameters, or training configuration\n"
            f"- New features, algorithms, or techniques being tested\n"
            f"- Bug fixes or improvements that impact training\n"
            f"- Data processing or environment changes\n\n"
            f"Ignore minor changes like formatting, comments, or documentation unless they're significant.\n\n"
            f"Git diff:\n"
            f"```\n{git_diff}\n```\n\n"
            f"Generate a brief, technical description suitable for a training run log:"
        )

        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=200, temperature=0.3, messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text.strip()

        except Exception as e:
            return f"Unable to generate description: API error: {str(e)}"
