from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module


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
