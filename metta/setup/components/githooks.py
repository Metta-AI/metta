import subprocess
import sys
from enum import Enum

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, green, info, prompt_choice


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
        return CommitHookMode.CHECK

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

    @property
    def setup_script_location(self) -> str | None:
        return "devops/setup_git_hooks.sh"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("githooks")

    def check_installed(self) -> bool:
        git_hooks_dir = self.repo_root / ".git" / "hooks"
        if not git_hooks_dir.exists():
            return False
        pre_commit = git_hooks_dir / "pre-commit"
        return pre_commit.exists() and pre_commit.is_file()

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

        # Run linting
        lint_cmd = ["metta", "lint", "--staged"]

        if hook_mode == CommitHookMode.FIX:
            lint_cmd.append("--fix")

        try:
            subprocess.run(lint_cmd, cwd=self.repo_root, check=True)

            # If in fix mode, stage the fixed files
            if hook_mode == CommitHookMode.FIX.value:
                subprocess.run(["git", "add"] + files, cwd=self.repo_root, check=True)
        except subprocess.CalledProcessError as e:
            if hook_mode == CommitHookMode.CHECK.value:
                error("Linting failed. Please fix the issues before committing.")
                error("Consider running `metta lint --fix` to fix some issues automatically.")
            sys.exit(e.returncode)
