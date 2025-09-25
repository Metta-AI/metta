import os
import subprocess
import sys
import tomllib
from enum import Enum
from pathlib import Path

from metta.common.util.fs import get_file_hash
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import colorize, error, info, prompt_choice, success


class CommitHookMode(Enum):
    NONE = "none"
    CHECK = "check"
    FIX = "fix"

    def get_description(self) -> str:
        descriptions = {
            CommitHookMode.NONE: "No pre-commit linting",
            CommitHookMode.CHECK: "Check only (fail if issues found)",
            CommitHookMode.FIX: "Auto-fix issues before committing",
        }
        return descriptions.get(self, self.value)

    @classmethod
    def get_default(cls) -> "CommitHookMode":
        return CommitHookMode.FIX

    @classmethod
    def parse(cls, value: str | None) -> "CommitHookMode":
        try:
            return CommitHookMode(value)
        except ValueError:
            return cls.get_default()


class GitLeaksMode(Enum):
    NONE = "none"
    CHECK = "check"
    BLOCK = "block"

    def get_description(self) -> str:
        descriptions = {
            GitLeaksMode.NONE: "No secrets scanning",
            GitLeaksMode.CHECK: "Scan and warn about secrets (non-blocking)",
            GitLeaksMode.BLOCK: "Scan and block commits with secrets",
        }
        return descriptions.get(self, self.value)

    @classmethod
    def get_default(cls) -> "GitLeaksMode":
        return GitLeaksMode.BLOCK

    @classmethod
    def parse(cls, value: str | None) -> "GitLeaksMode":
        try:
            return GitLeaksMode(value)
        except ValueError:
            return cls.get_default()


@register_module
class GitHooksSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "Git hooks"

    def _is_applicable(self) -> bool:
        return not self._is_in_worktree()

    def _is_in_worktree(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip() == "true"
        except subprocess.CalledProcessError:
            return False

    def _get_main_repo_root(self) -> Path:
        """Get the main repository root, even if we're in a worktree"""
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).parent

    def _get_hooks_paths(self) -> tuple[Path, Path]:
        main_repo_root = self._get_main_repo_root()
        hooks_source_dir = self.repo_root / "devops" / "git-hooks"
        git_hooks_dir = main_repo_root / ".git" / "hooks"
        return hooks_source_dir, git_hooks_dir

    def _get_hook_files(self, hooks_source_dir: Path) -> list[Path]:
        if not hooks_source_dir.exists():
            return []
        return [
            hook_file
            for hook_file in hooks_source_dir.iterdir()
            if hook_file.is_file() and not hook_file.name.startswith(".")
        ]

    def check_installed(self) -> bool:
        """Check if all hooks from devops/git-hooks are installed with matching content"""
        hooks_source_dir, git_hooks_dir = self._get_hooks_paths()

        for hook_file in self._get_hook_files(hooks_source_dir):
            installed_hook = git_hooks_dir / hook_file.name

            # Check if the hook exists
            if not installed_hook.exists():
                return False

            if get_file_hash(hook_file) != get_file_hash(installed_hook):
                return False
        return True

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        """Install git hooks by symlinking from devops/git-hooks to .git/hooks"""
        # Check if we're in a worktree
        if self._is_in_worktree():
            info("Cannot install git hooks from a worktree. Skipping...")
            return

        info("Installing git hooks...")

        hooks_source_dir, git_hooks_dir = self._get_hooks_paths()
        git_hooks_dir.mkdir(exist_ok=True, parents=True)

        # Install each hook
        hook_count = 0
        for hook_file in self._get_hook_files(hooks_source_dir):
            target_hook = git_hooks_dir / hook_file.name
            if target_hook.exists() or target_hook.is_symlink():
                target_hook.unlink()

            # Create symlink
            target_hook.symlink_to(hook_file.resolve())

            # Make sure the source is executable
            hook_file.chmod(hook_file.stat().st_mode | 0o111)

            hook_count += 1
            info(f"  Installed: {hook_file.name}")

        success(f"Successfully installed {colorize(str(hook_count), 'green')} git hooks")

    def get_configuration_options(self) -> dict[str, tuple[str, str]]:
        return {
            "commit_hook_mode": (CommitHookMode.CHECK.value, "Pre-commit hook behavior"),
            "gitleaks_mode": (GitLeaksMode.BLOCK.value, "Gitleaks secrets scanning behavior"),
        }

    def configure(self) -> None:
        info("Configuring git commit hooks...")

        current_commit = CommitHookMode.parse(self.get_setting("commit_hook_mode", default=None))
        current_gitleaks = GitLeaksMode.parse(self.get_setting("gitleaks_mode", default=None))

        # Prompt for new modes
        if os.environ.get("METTA_TEST_ENV") or os.environ.get("CI"):
            commit_mode = CommitHookMode.get_default()
            gitleaks_mode = GitLeaksMode.get_default()
        else:
            commit_mode = prompt_choice(
                "Select pre-commit hook behavior:",
                [(mode, mode.get_description()) for mode in CommitHookMode],
                default=CommitHookMode.get_default(),
                current=current_commit,
            )

            gitleaks_mode = prompt_choice(
                "Select gitleaks secrets scanning behavior:",
                [(mode, mode.get_description()) for mode in GitLeaksMode],
                default=GitLeaksMode.get_default(),
                current=current_gitleaks,
            )

        # Save the settings
        self.set_setting("commit_hook_mode", commit_mode.value)
        self.set_setting("gitleaks_mode", gitleaks_mode.value)

        if commit_mode.value == CommitHookMode.CHECK.value:
            info("Using default mode: check only")
        else:
            info(f"Commit hook mode set to: {colorize(commit_mode.get_description(), 'green')}")

        info(f"Gitleaks mode set to: {colorize(gitleaks_mode.get_description(), 'green')}")

    def _check_gitleaks_installed(self) -> bool:
        try:
            subprocess.run(
                ["gitleaks", "version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_gitleaks(self, gitleaks_mode: GitLeaksMode) -> bool:
        if gitleaks_mode == GitLeaksMode.NONE:
            return True

        if not self._check_gitleaks_installed():
            info("Gitleaks not installed. Install with: brew install gitleaks")
            info("Skipping secrets scanning...")
            return True

        try:
            subprocess.run(
                ["gitleaks", "protect", "--staged", "-v", "--no-banner", "--exit-code", "1"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            if e.stdout:
                info(e.stdout)
            if e.stderr:
                error(e.stderr)

            error("Review the output above and remove any secrets before committing.")

            if gitleaks_mode == GitLeaksMode.BLOCK:
                error("Commit blocked due to detected secrets.")
                return False
            else:
                info("Warning: Proceeding despite detected secrets (check mode).")
                return True

    def run(self, args: list[str]) -> None:
        if not args or args[0] != "pre-commit":
            error("Usage: metta run githooks pre-commit")
            sys.exit(1)

        hook_mode = CommitHookMode.parse(self.get_setting("commit_hook_mode", default=None))
        gitleaks_mode = GitLeaksMode.parse(self.get_setting("gitleaks_mode", default=None))

        # Run gitleaks check first
        if not self._run_gitleaks(gitleaks_mode):
            sys.exit(1)

        if hook_mode == CommitHookMode.NONE:
            sys.exit(0)

        # Get staged Python files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f]

        if not files:
            # No Python files to lint
            sys.exit(0)

        # Filter out excluded paths based on .ruff.toml
        ruff_config_path = self.repo_root / ".ruff.toml"
        with open(ruff_config_path, "rb") as f:
            config = tomllib.load(f)
        excluded_paths = config.get("exclude", [])

        if excluded_paths:
            filtered_files = []
            for f in files:
                file_path = Path(f)
                should_exclude = any(
                    file_path == Path(excluded) or file_path.is_relative_to(Path(excluded))
                    for excluded in excluded_paths
                )
                if not should_exclude:
                    filtered_files.append(f)

            files = filtered_files

        if not files:
            # No files to lint after exclusions
            sys.exit(0)

        lint_cmd = ["metta", "lint"]
        if hook_mode == CommitHookMode.FIX:
            lint_cmd.append("--fix")
        lint_cmd.extend(files)

        try:
            subprocess.run(lint_cmd, cwd=self.repo_root, check=True)

            if hook_mode == CommitHookMode.FIX:
                subprocess.run(["git", "add"] + files, cwd=self.repo_root, check=True)

        except subprocess.CalledProcessError as e:
            if hook_mode == CommitHookMode.CHECK:
                error("Linting failed. Please fix the issues before committing.")
                error("Consider running `metta lint --fix` to fix some issues automatically.")
            sys.exit(e.returncode)
