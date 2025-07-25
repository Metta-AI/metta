import subprocess
import sys
from enum import Enum
from pathlib import Path

from metta.common.util.fs import get_file_hash
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, green, info, prompt_choice, success


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


@register_module
class GitHooksSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "Git hooks"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("githooks") and not self._is_in_worktree()

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

    def install(self) -> None:
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

        success(f"Successfully installed {green(str(hook_count))} git hooks")

    def get_configuration_options(self) -> dict[str, tuple[str, str]]:
        return {
            "commit_hook_mode": (CommitHookMode.CHECK.value, "Pre-commit hook behavior"),
        }

    def configure(self) -> None:
        info("Configuring git commit hooks...")

        current = CommitHookMode.parse(self.get_setting("commit_hook_mode", default=None))

        # Prompt for new mode
        mode = prompt_choice(
            "Select pre-commit hook behavior:",
            [(mode, mode.get_description()) for mode in CommitHookMode],
            default=CommitHookMode.get_default(),
            current=current,
        )

        # Save the setting (only if non-default)
        self.set_setting("commit_hook_mode", mode.value)

        if mode.value == CommitHookMode.CHECK.value:
            info("Using default mode: check only")
        else:
            print(f"Commit hook mode set to: {green(mode.get_description())}")

    def run(self, args: list[str]) -> None:
        if not args or args[0] != "pre-commit":
            error("Usage: metta run githooks pre-commit")
            sys.exit(1)

        hook_mode = CommitHookMode.parse(self.get_setting("commit_hook_mode", default=None))

        if hook_mode == CommitHookMode.NONE:
            sys.exit(0)

        # Get staged files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        all_files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        python_files = [f for f in all_files if f.endswith(".py")]
        notebook_files = [f for f in all_files if f.endswith(".ipynb") and "/log/" in f]

        if not python_files and not notebook_files:
            # No files to process
            sys.exit(0)

        # Process Python files
        if python_files:
            lint_cmd = ["metta", "lint", "--staged"]

            if hook_mode == CommitHookMode.FIX:
                lint_cmd.append("--fix")

            try:
                subprocess.run(lint_cmd, cwd=self.repo_root, check=True)

                # If in fix mode, stage the fixed files
                if hook_mode == CommitHookMode.FIX:
                    subprocess.run(["git", "add"] + python_files, cwd=self.repo_root, check=True)
            except subprocess.CalledProcessError as e:
                if hook_mode == CommitHookMode.CHECK:
                    error("Linting failed. Please fix the issues before committing.")
                    error("Consider running `metta lint --fix` to fix some issues automatically.")
                sys.exit(e.returncode)

        # Strip notebook outputs from log directory
        if notebook_files:
            info(f"Stripping outputs from {len(notebook_files)} notebook(s) in log/...")
            for notebook in notebook_files:
                try:
                    # Strip outputs using nbstripout
                    subprocess.run(["nbstripout", notebook], cwd=self.repo_root, check=True)
                    # Stage the stripped notebook
                    subprocess.run(["git", "add", notebook], cwd=self.repo_root, check=True)
                    info(f"  ✓ Stripped: {notebook}")
                except subprocess.CalledProcessError:
                    error(f"Failed to strip outputs from {notebook}")
                    sys.exit(1)
                except FileNotFoundError:
                    error("nbstripout not found. Please install it: pip install nbstripout")
                    sys.exit(1)
