import os
import subprocess
import sys
from enum import Enum
from pathlib import Path

from metta.common.util.fs import get_file_hash
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.tools.linting import get_staged_files
from metta.setup.utils import colorize, error, info, prompt_choice, success, warning


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

    def _apply_skip_hooks(self, env: dict[str, str], hooks_to_skip: list[str]) -> None:
        if not hooks_to_skip:
            return

        existing = env.get("SKIP", "")
        parts = [item.strip() for item in existing.split(",") if item.strip()]
        parts.extend(hooks_to_skip)
        deduped = list(dict.fromkeys(parts))
        env["SKIP"] = ",".join(deduped)

    def _run_metta_lint(self, mode: CommitHookMode, env: dict[str, str]) -> subprocess.CompletedProcess:
        cmd = ["uv", "run", "--active", "metta", "lint", "--staged"]
        if mode == CommitHookMode.FIX:
            cmd.append("--fix")
        else:
            cmd.append("--check")
        return subprocess.run(cmd, cwd=self.repo_root, env=env, check=False)

    def _run_detect_private_key(self, stage: str, files: list[str], warn_only: bool) -> bool:
        cmd = ["uv", "run", "--active", "pre-commit", "run", "detect-private-key", "--hook-stage", stage]
        if files:
            cmd.extend(["--files", *files])
        else:
            cmd.append("--all-files")

        result = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True

        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip()

        if warn_only:
            if output:
                warning(output)
            warning("Secrets detected by detect-private-key (non-blocking mode).")
            return False

        if output:
            error(output)
        else:
            error("detect-private-key failed.")
        return False

    def run(self, args: list[str]) -> None:
        if not args or args[0] != "pre-commit":
            error("Usage: metta run githooks pre-commit")
            sys.exit(1)

        hook_mode = CommitHookMode.parse(self.get_setting("commit_hook_mode", default=None))
        gitleaks_mode = GitLeaksMode.parse(self.get_setting("gitleaks_mode", default=None))

        staged_files = get_staged_files(self.repo_root)

        skip_hooks: list[str] = []
        warn_on_secrets = False
        if gitleaks_mode == GitLeaksMode.NONE:
            skip_hooks.append("detect-private-key")
        elif gitleaks_mode == GitLeaksMode.CHECK:
            skip_hooks.append("detect-private-key")
            warn_on_secrets = True

        env = os.environ.copy()
        self._apply_skip_hooks(env, skip_hooks)

        if hook_mode == CommitHookMode.NONE:
            if gitleaks_mode == GitLeaksMode.BLOCK:
                success_detect = self._run_detect_private_key("pre-commit", staged_files, warn_only=False)
                sys.exit(0 if success_detect else 1)

            if gitleaks_mode == GitLeaksMode.CHECK:
                self._run_detect_private_key("manual", staged_files, warn_only=True)
            sys.exit(0)

        result = self._run_metta_lint(hook_mode, env)
        if result.returncode != 0:
            if hook_mode == CommitHookMode.CHECK:
                error("Linting failed. Please fix the issues before committing.")
                error("Consider running `metta lint --fix` to fix some issues automatically.")
            sys.exit(result.returncode)

        if warn_on_secrets:
            refreshed_staged = get_staged_files(self.repo_root)
            self._run_detect_private_key("manual", refreshed_staged, warn_only=True)
