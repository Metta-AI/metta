#!/usr/bin/env -S uv run
# need this to import and call suppress_noisy_logs first
# ruff: noqa: E402
from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()
import concurrent.futures
import re
import subprocess
import sys
import webbrowser
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import gitta as git
from metta.common.util.fs import get_repo_root
from metta.common.util.log_config import init_logging
from metta.setup.components.base import SetupModuleStatus
from metta.setup.local_commands import app as local_app
from metta.setup.tools.book import app as book_app
from metta.setup.tools.ci_runner import cmd_ci
from metta.setup.tools.clean import cmd_clean
from metta.setup.tools.code_formatters import app as code_formatters_app
from metta.setup.tools.test_runner.test_cpp import app as cpp_test_runner_app
from metta.setup.tools.test_runner.test_nim import app as nim_test_runner_app
from metta.setup.tools.test_runner.test_python import app as python_test_runner_app
from metta.setup.utils import debug, error, info, success, warning
from metta.utils.live_run_monitor import app as run_monitor_app
from softmax.dashboard.report import app as softmax_system_health_app

if TYPE_CHECKING:
    from metta.setup.registry import SetupModule

VERSION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+(?:\.\d+)?)$")
DEFAULT_INITIAL_VERSION = "0.0.0.1"


class PRStatus(StrEnum):
    """GitHub PR status filter options."""

    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    ALL = "all"


class MettaCLI:
    def __init__(self):
        self._components_initialized = False

    def _init_all(self):
        """Initialize all components - used by commands that need everything."""
        if self._components_initialized:
            return

        from metta.setup.utils import import_all_modules_from_subpackage

        import_all_modules_from_subpackage("metta.setup", "components")
        self._components_initialized = True

    def setup_wizard(self, non_interactive: bool = False):
        from metta.setup.profiles import UserType
        from metta.setup.saved_settings import get_saved_settings
        from metta.setup.utils import header, info, prompt_choice, success

        header("Welcome to Metta!\n\n")
        info("Note: You can run 'metta configure <component>' to change component-level settings later.\n")

        saved_settings = get_saved_settings()
        if saved_settings.exists():
            info("Current configuration:")
            info(f"Profile: {saved_settings.user_type.value}")
            info(f"Mode: {'custom' if saved_settings.is_custom_config else 'profile'}")
            info("\nEnabled components:")
            components = saved_settings.get_components()
            for comp, settings in components.items():
                if settings.get("enabled"):
                    success(f"  + {comp}")
            info("\n")

        choices = [(ut, ut.get_description()) for ut in UserType]

        current_user_type = saved_settings.user_type if saved_settings.exists() else None

        result = prompt_choice(
            "Select configuration:",
            choices,
            current=current_user_type,
            non_interactive=non_interactive,
        )

        if result == UserType.CUSTOM:
            self._custom_setup(non_interactive=non_interactive)
        else:
            saved_settings.apply_profile(result)
            success(f"\nConfigured as {result.value} user.")
        info("\nRun 'metta install' to set up your environment.")

    def _custom_setup(self, non_interactive: bool = False):
        from metta.setup.profiles import PROFILE_DEFINITIONS, UserType
        from metta.setup.registry import get_all_modules
        from metta.setup.saved_settings import get_saved_settings
        from metta.setup.utils import prompt_choice

        user_type = prompt_choice(
            "Select base profile for custom configuration:",
            [(ut, ut.get_description()) for ut in UserType if ut != UserType.CUSTOM],
            default=UserType.EXTERNAL,
            non_interactive=non_interactive,
        )

        saved_settings = get_saved_settings()
        saved_settings.setup_custom_profile(user_type)

        info("\nCustomize components:")
        all_modules = get_all_modules()

        for module in all_modules:
            current_enabled = saved_settings.is_component_enabled(module.name)

            enabled = prompt_choice(
                f"Enable {module.name} ({module.description})?",
                [(True, "Yes"), (False, "No")],
                default=current_enabled,
                current=current_enabled,
                non_interactive=non_interactive,
            )

            profile_default = (
                PROFILE_DEFINITIONS.get(user_type, {}).get("components", {}).get(module.name, {}).get("enabled", False)
            )
            if enabled != profile_default:
                saved_settings.set(f"components.{module.name}.enabled", enabled)

        success("\nCustom configuration saved.")
        info("\nRun 'metta install' to set up your environment.")

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."


cli = MettaCLI()
app = typer.Typer(
    help="Metta Setup Tool - Configure and install development environment",
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _ensure_components_initialized() -> None:
    """Load setup components on demand for commands that need them."""
    cli._init_all()


def _bump_version(version: str) -> str:
    parts = version.split(".")
    bumped = parts[:-1] + [str(int(parts[-1]) + 1)]
    return ".".join(bumped)


def _validate_version_format(version: str) -> None:
    if not VERSION_PATTERN.match(version):
        error(f"Invalid version '{version}'. Expected numeric segments like '1.2.3' or '1.2.3.4'.")
        raise typer.Exit(1)


def _ensure_tag_unique(package: str, version: str) -> None:
    tag_name = f"{package}-v{version}"
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}"],
        cwd=get_repo_root(),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        error(f"Tag '{tag_name}' already exists.")
        raise typer.Exit(1)


@app.command(name="configure", help="Configure Metta settings")
def cmd_configure(
    component: Annotated[Optional[str], typer.Argument(help="Specific component to configure")] = None,
    profile: Annotated[
        Optional[str],
        typer.Option(
            "--profile",
            help="Set user profile",
        ),
    ] = None,
    non_interactive: Annotated[bool, typer.Option("--non-interactive", help="Non-interactive mode")] = False,
):
    """Configure Metta settings."""
    if component:
        if profile:
            error("Cannot configure a component and a profile at the same time.")
            raise typer.Exit(1)
        configure_component(component)
    elif profile:
        from metta.setup.profiles import PROFILE_DEFINITIONS, UserType
        from metta.setup.saved_settings import get_saved_settings

        selected_user_type = UserType(profile)
        if selected_user_type in PROFILE_DEFINITIONS:
            saved_settings = get_saved_settings()
            saved_settings.apply_profile(selected_user_type)
            success(f"Configured as {selected_user_type.value} user.")
        else:
            error(f"Unknown profile: {profile}")
            raise typer.Exit(1)
    else:
        cli.setup_wizard(non_interactive=non_interactive)


def configure_component(component_name: str):
    _ensure_components_initialized()
    from metta.setup.registry import get_all_modules
    from metta.setup.utils import error, info

    modules = get_all_modules()
    module_map = {m.name: m for m in modules}

    if not (module := module_map.get(component_name)):
        error(f"Unknown component: {component_name}")
        info(f"Available components: {', '.join(sorted(module_map.keys()))}")
        raise typer.Exit(1)

    options = module.get_configuration_options()
    if not options:
        info(f"Component '{component_name}' has no configuration options.")
        return
    module.configure()


def _get_selected_modules(components: list[str] | None = None, ensure_required: bool = False) -> list["SetupModule"]:
    _ensure_components_initialized()
    from metta.setup.registry import get_all_modules

    component_objs = [
        m
        for m in get_all_modules()
        if (
            (components is not None and (m.name in components) or (ensure_required and m.always_required))
            or (components is None and m.is_enabled())
        )
    ]
    if components:
        component_names = {m.name for m in component_objs}
        not_found_components = [c for c in components if c not in component_names]
        if not_found_components:
            error(f"Unknown components: {', '.join(not_found_components)}")
            raise typer.Exit(1)
    return component_objs


def _get_all_package_names() -> list[str]:
    """Return all valid package names in <repo_root>/packages/ (dirs with __init__.py or setup.py)."""
    packages_dir = get_repo_root() / "packages"
    if not packages_dir.exists():
        return []
    return sorted(p.name for p in packages_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


@app.command(name="install", help="Install or update components")
def cmd_install(
    components: Annotated[Optional[list[str]], typer.Argument(help="Components to install")] = None,
    profile: Annotated[Optional[str], typer.Option("--profile", help="Profile to configure before installing")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force reinstall")] = False,
    no_clean: Annotated[bool, typer.Option("--no-clean", help="Skip cleaning before install")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive", help="Non-interactive mode")] = False,
    check_status: Annotated[bool, typer.Option("--check-status", help="Check status after installation")] = True,
):
    if not no_clean:
        cmd_clean(force=force)

    from metta.setup.saved_settings import get_saved_settings

    # A profile must exist before installing. If installing in non-interactive mode,
    # the target profile must be specified with --profile. If in interactive mode and
    # no profile is specified, the setup wizard will be run.
    profile_exists = get_saved_settings().exists()
    if non_interactive and not profile_exists and not profile:
        error("Must specify a profile if installing in non-interactive mode without an existing one.")
        raise typer.Exit(1)
    elif profile or not profile_exists:
        cmd_configure(profile=profile, non_interactive=non_interactive, component=None)

    always_required_components = ["uv", "system"]
    modules = _get_selected_modules((always_required_components + components) if components else None)

    if not modules:
        info("No modules to install.")
        return

    info(f"\nInstalling {len(modules)} components...\n")

    for module in modules:
        force_install = force and (components is None) or (components is not None and module.name in components)
        info(f"[{module.name}] {module.description}" + (" (force install)" if force_install else ""))

        if module.install_once and module.check_installed() and not force_install:
            debug("  -> Already installed, skipping (use --force to reinstall)\n")
            continue

        try:
            module.install(non_interactive=non_interactive, force=force_install)
            print()
        except Exception as e:
            error(f"  Error: {e}\n")

    if not non_interactive and check_status:
        cmd_status(components=components, non_interactive=non_interactive)


@app.command(name="status", help="Show status of components")
def cmd_status(
    components: Annotated[
        Optional[list[str]],
        typer.Option("--components", help="Comma-separated list of components. Defaults to all enabled components."),
    ] = None,
    non_interactive: Annotated[bool, typer.Option("-n", "--non-interactive", help="Non-interactive mode")] = False,
):
    modules = _get_selected_modules(components if components else None, ensure_required=True)
    if not modules:
        warning("No modules to check.")
        return

    modules_by_name = {m.name: m for m in modules}
    module_status: dict[str, SetupModuleStatus] = {}

    console = Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Checking component status...", total=None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_module = {executor.submit(lambda m: (m.name, m.get_status()), m): m for m in modules}
            for future in concurrent.futures.as_completed(future_to_module):
                name, status = future.result()
                if status:
                    module_status[name] = status

    table = Table(title="Component Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Installed", justify="center")
    table.add_column("Connected As")
    table.add_column("Expected")
    table.add_column("Status", justify="center")

    for module in modules:
        if module.name not in module_status:
            continue

        status_data = module_status[module.name]
        installed = status_data.installed
        connected_as = status_data.connected_as
        expected = status_data.expected

        installed_str = "Yes" if installed else "No"
        connected_str = cli._truncate(connected_as or "-", 25)
        expected_str = cli._truncate(expected or "-", 20)

        if not installed:
            status = "[red]NOT INSTALLED[/red]"
        elif connected_as is None:
            if expected is None:
                status = "[green]OK[/green]"
            else:
                status = "[red]NOT CONNECTED[/red]"
        elif expected is None:
            status = "[green]OK[/green]"
        elif expected in connected_as:
            status = "[green]OK[/green]"
        else:
            status = "[yellow]WRONG ACCOUNT[/yellow]"

        table.add_row(module.name, installed_str, connected_str, expected_str, status)

    console = Console()
    console.print(table)

    from metta.tools.utils.auto_config import auto_policy_storage_decision

    policy_decision = auto_policy_storage_decision()
    if policy_decision.using_remote and policy_decision.base_prefix:
        if policy_decision.reason == "env_override":
            success(
                f"Policy storage: S3 uploads enabled via POLICY_REMOTE_PREFIX → {policy_decision.base_prefix}/<run>."
            )
        else:
            success(f"Policy storage: Softmax S3 uploads active → {policy_decision.base_prefix}/<run>.")
    elif policy_decision.reason == "not_connected" and policy_decision.base_prefix:
        warning(
            "Policy storage: local only. Run 'aws sso login --profile softmax' to enable uploads to "
            f"{policy_decision.base_prefix}/<run>."
        )
    elif policy_decision.reason == "aws_not_enabled":
        info("Policy storage: local only (AWS component disabled).")
    elif policy_decision.reason == "no_base_prefix":
        info(
            "Policy storage: local only (remote policy prefix not configured). "
            "Set POLICY_REMOTE_PREFIX or rerun 'metta configure aws'."
        )
    could_force_install = [
        name
        for name, data in module_status.items()
        if (
            not data.installed  # Not installed
            or (  # Expected to be connected as a specific account, but is not, and can remediate through force install
                data.expected is not None
                and data.connected_as != data.expected
                and modules_by_name[name].can_remediate_connected_status_with_install
            )
        )
    ]
    if could_force_install and not non_interactive and sys.stdin.isatty():
        if typer.confirm(f"\nForce install {', '.join(could_force_install)} to attempt to resolve issues?"):
            cmd_install(components=could_force_install, non_interactive=non_interactive, force=True, check_status=False)


@app.command(name="run", help="Run component-specific commands")
def cmd_run(
    component: Annotated[str, typer.Argument(help="Component to run command for")],
    args: Annotated[Optional[list[str]], typer.Argument(help="Arguments to pass to the component")] = None,
):
    _ensure_components_initialized()
    from metta.setup.registry import get_all_modules

    modules = get_all_modules()
    module_map = {m.name: m for m in modules}

    if not (module := module_map.get(component)):
        error(f"Unknown component: {component}")
        info(f"Available components: {', '.join(sorted(module_map.keys()))}")
        raise typer.Exit(1)

    module.run(args or [])


@app.command(name="clean", help="Clean build artifacts and temporary files")
def clean(
    force: Annotated[bool, typer.Option("--force", help="Force clean")] = False,
):
    cmd_clean(force=force)


@app.command(name="publish", help="Create and push a release tag for a package")
def cmd_publish(
    package: Annotated[str, typer.Argument(help="Package to publish")],
    version_override: Annotated[
        Optional[str],
        typer.Option("--version", "-v", help="Explicit version to tag (digits separated by dots)"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview actions without tagging")] = False,
    no_repo: Annotated[bool, typer.Option("--no-repo", help="Don't push the github repo")] = False,
    remote: Annotated[str, typer.Option("--remote", help="Git remote to push the tag to")] = "origin",
    force: Annotated[bool, typer.Option("--force", help="Bypass branch and clean checks")] = False,
):
    package = package.lower()
    if package not in _get_all_package_names():
        error(f"Unsupported package '{package}'. Supported packages: {', '.join(sorted(_get_all_package_names()))}.")
        raise typer.Exit(1)

    prefix = f"{package}-v"
    try:
        info(f"Fetching tags from {remote}...")
        git.run_git("fetch", remote, "--tags", "--force")
    except subprocess.CalledProcessError as exc:
        error(f"Failed to fetch tags from {remote}: {exc}")
        raise typer.Exit(exc.returncode) from exc

    try:
        status_output = git.run_git("status", "--porcelain")
    except subprocess.CalledProcessError as exc:
        error(f"Failed to read git status: {exc}")
        raise typer.Exit(exc.returncode) from exc

    if status_output.strip() and not force:
        error("Working tree is not clean. Commit, stash, or clean changes before publishing (use --force to override).")
        raise typer.Exit(1)

    try:
        current_branch = git.run_git("rev-parse", "--abbrev-ref", "HEAD")
        current_commit = git.run_git("rev-parse", "HEAD")
    except subprocess.CalledProcessError as exc:
        error(f"Failed to determine git state: {exc}")
        raise typer.Exit(exc.returncode) from exc

    if current_branch not in {"main"} and not force:
        error("Publishing is only supported from the main branch. Switch to 'main' or pass --force to override.")
        raise typer.Exit(1)

    try:
        tag_list_output = git.run_git("tag", "--list", f"{prefix}*", "--sort=-v:refname")
    except subprocess.CalledProcessError:
        tag_list_output = ""

    tags = [line for line in tag_list_output.splitlines() if line.strip()]
    latest_tag = tags[0] if tags else None

    if version_override is None:
        if latest_tag:
            previous_version = latest_tag[len(prefix) :]
            _validate_version_format(previous_version)
            target_version = _bump_version(previous_version)
        else:
            target_version = DEFAULT_INITIAL_VERSION
    else:
        _validate_version_format(version_override)
        target_version = version_override

    _validate_version_format(target_version)
    _ensure_tag_unique(package, target_version)

    tag_name = f"{prefix}{target_version}"

    info("Release summary:\n")
    info(f"  Package: {package}")
    info(f"  Current branch: {current_branch}")
    info(f"  Commit: {current_commit}")
    info(f"  Tag: {tag_name}")
    if latest_tag:
        info(f"  Previous tag: {latest_tag}")
    else:
        info("  Previous tag: none")
    if force:
        warning("Force mode enabled: branch and clean checks were bypassed.")
    info("")

    publish_mettagrid_after = False

    if dry_run:
        success("Dry run: no tag created. Run without --dry-run to proceed.")
        return

    if package == "cogames":
        publish_mettagrid_after = typer.confirm(
            "Cogames depends on mettagrid. Publish mettagrid after this tag?",
            default=True,
        )

    if not typer.confirm("Create and push this tag?", default=True):
        info("Publishing aborted.")
        return

    try:
        info(f"Tagging {package} {target_version}...")
        git.run_git("tag", "-a", tag_name, "-m", f"Release {package} {target_version}")
        git.run_git("push", remote, tag_name)
    except subprocess.CalledProcessError as exc:
        error(f"Failed to tag: {exc}.")
        raise typer.Exit(exc.returncode) from exc

    success(f"Published {tag_name} to {remote}.")

    try:
        if not no_repo:
            info(f"Pushing {package} as child repo...")
            subprocess.run([f"{get_repo_root()}/devops/git/push_child_repo.py", package, "-y"], check=True)
    except subprocess.CalledProcessError as exc:
        warning(
            f"Failed to push child repo: {exc}. {tag_name} was still published to {remote}."
            + " Use --no-repo to skip pushing to github repo."
        )

    if publish_mettagrid_after:
        info("")
        info("Starting mettagrid publish flow...")
        cmd_publish(
            package="mettagrid",
            version_override=None,
            dry_run=False,
            no_repo=no_repo,
            remote=remote,
            force=force,
        )


@app.command(name="tool", help="Run a tool from the tools/ directory", context_settings={"allow_extra_args": True})
def cmd_tool(
    tool_name: Annotated[str, typer.Argument(help="Name of the tool to run")],
    ctx: typer.Context,
):
    tool_path = get_repo_root() / "tools" / f"{tool_name}.py"
    if not tool_path.exists():
        error(f"Error: Tool '{tool_name}' not found at {tool_path}")
        raise typer.Exit(1)

    cmd = [str(tool_path)] + (ctx.args or [])
    try:
        subprocess.run(cmd, cwd=get_repo_root(), check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode) from e


@app.command(name="shell", help="Start an IPython shell with Metta imports")
def cmd_shell():
    cmd = ["uv", "run", "--active", "metta/setup/shell.py"]
    try:
        subprocess.run(cmd, cwd=get_repo_root(), check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode) from e


@app.command(name="go", help="Navigate to a Softmax Home shortcut", context_settings={"allow_extra_args": True})
def cmd_go(ctx: typer.Context):
    if not ctx.args:
        error("Please specify a shortcut (e.g., 'metta go g' for GitHub)")
        info("\nCommon shortcuts:")
        info("  g    - GitHub")
        info("  w    - Weights & Biases")
        info("  o    - Observatory")
        info("  d    - Datadog")
        return

    shortcut = ctx.args[0]
    url = f"https://home.softmax-research.net/{shortcut}"

    info(f"Opening {url}...")
    webbrowser.open(url)


@app.command(name="pr-feed", help="Show PRs that touch a specific path")
def cmd_pr_feed(
    path: Annotated[str, typer.Argument(help="Path filter (e.g., metta/jobs)")],
    status: Annotated[PRStatus, typer.Option("--status", help="PR status filter")] = PRStatus.OPEN,
    num_days: Annotated[int, typer.Option("--num_days", help="Search PRs updated in last N days")] = 30,
):
    """Show PRs that touch files in a specific path.

    Automatically fetches all pages within the time window.
    """
    import json
    from datetime import datetime, timedelta, timezone

    # Convert status to GraphQL enum
    # Note: "closed" includes both CLOSED (without merge) and MERGED
    status_mapping = {
        PRStatus.OPEN: ("OPEN", "OPEN"),
        PRStatus.CLOSED: ("CLOSED, MERGED", "CLOSED/MERGED"),
        PRStatus.MERGED: ("MERGED", "MERGED"),
        PRStatus.ALL: ("OPEN, CLOSED, MERGED", "ALL"),
    }
    states, status_display = status_mapping[status]

    console = Console()
    console.print(
        f"🔍 Searching for [yellow]{status_display}[/yellow] PRs touching: [cyan]{path}[/cyan] (last {num_days} days)"
    )
    console.print()

    # Calculate cutoff date
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=num_days)

    try:
        cursor = None
        total_found = 0
        page_num = 0

        while True:
            page_num += 1
            # Build pagination parameter
            after_clause = f', after: "{cursor}"' if cursor else ""

            # Run GraphQL query via gh CLI (ordered by most recently updated)
            query = f"""
            query {{
              repository(owner: "Metta-AI", name: "metta") {{
                pullRequests(
                  first: 100,
                  states: [{states}],
                  orderBy: {{field: UPDATED_AT, direction: DESC}}{after_clause}
                ) {{
                  pageInfo {{
                    hasNextPage
                    endCursor
                  }}
                  nodes {{
                    number
                    title
                    url
                    author {{ login }}
                    updatedAt
                    files(first: 100) {{
                      nodes {{ path }}
                    }}
                  }}
                }}
              }}
            }}
            """

            result = subprocess.run(
                ["gh", "api", "graphql", "-f", f"query={query}"],
                capture_output=True,
                text=True,
                check=True,
            )

            data = json.loads(result.stdout)
            pull_requests = data["data"]["repository"]["pullRequests"]
            prs = pull_requests["nodes"]
            page_info = pull_requests["pageInfo"]

            # Filter and display PRs that touch the specified path
            page_matches = 0
            oldest_pr_date = None

            for pr in prs:
                # Parse update date
                updated = datetime.fromisoformat(pr["updatedAt"].replace("Z", "+00:00"))
                oldest_pr_date = updated  # Track oldest in this batch

                # Skip if older than cutoff
                if updated < cutoff_date:
                    break

                # Check if touches our path
                if any(file["path"].startswith(path) for file in pr["files"]["nodes"]):
                    page_matches += 1
                    total_found += 1
                    updated_str = updated.strftime("%Y-%m-%d")

                    console.print(f"[green]PR #{pr['number']}:[/green] {pr['title']}")
                    console.print(f"  Author: [cyan]@{pr['author']['login']}[/cyan] • Updated: {updated_str}")
                    console.print(f"  {pr['url']}")
                    console.print()

            # Check if we should continue
            if not page_info["hasNextPage"]:
                break

            # Stop if we've gone past the cutoff date
            if oldest_pr_date and oldest_pr_date < cutoff_date:
                break

            # Continue to next page
            cursor = page_info["endCursor"]

        # Summary
        if total_found == 0:
            console.print(f"[yellow]No {status_display} PRs found touching {path} in the last {num_days} days[/yellow]")
        else:
            console.print(f"[green]✅ Found {total_found} {status_display} PR(s)[/green]")

    except subprocess.CalledProcessError as e:
        error(f"Failed to fetch PRs: {e.stderr}")
        raise typer.Exit(1) from e
    except Exception as e:
        error(f"Error: {e}")
        raise typer.Exit(1) from e


# Report env details command
@app.command(name="report-env-details", help="Report environment details including UV project directory")
def cmd_report_env_details():
    """Report environment details."""
    info(f"UV Project Directory: {get_repo_root()}")
    info(f"Metta CLI Working Directory: {Path.cwd()}")
    if branch := git.get_current_branch():
        info(f"Git Branch: {branch}")
    if commit := git.get_current_commit():
        info(f"Git Commit: {commit}")


@app.command(
    name="clip",
    help="Copy codebase to clipboard. Pass through any codeclip flags",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,  # Disable typer's help handling
)
def cmd_clip(
    ctx: typer.Context,
):
    """Copy subsets of codebase for LLM contexts."""
    # Find all arguments after 'clip' command
    clip_index = sys.argv.index("clip")
    args_after_clip = sys.argv[clip_index + 1 :]

    # Build command with codeclip and pass all arguments through
    cmd = ["codeclip"] + args_after_clip

    try:
        subprocess.run(cmd, cwd=get_repo_root(), check=False)
    except FileNotFoundError:
        error("Error: Command not found: codeclip")
        info("Run: metta install codebot")
        raise typer.Exit(1) from None


@app.command(name="gridworks", help="Start the Gridworks web UI", context_settings={"allow_extra_args": True})
def cmd_gridworks(ctx: typer.Context):
    cmd = ["./gridworks/start.py", *ctx.args]
    subprocess.run(cmd, cwd=get_repo_root(), check=False)


app.add_typer(run_monitor_app, name="run-monitor", help="Monitor training runs.")
app.add_typer(local_app, name="local")
app.add_typer(book_app, name="book")
app.add_typer(softmax_system_health_app, name="softmax-system-health")
app.add_typer(python_test_runner_app, name="pytest")
app.add_typer(cpp_test_runner_app, name="cpptest")
app.add_typer(nim_test_runner_app, name="nimtest")
app.command(
    name="ci",
    help="Run CI checks locally",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True, "allow_interspersed_args": False},
)(cmd_ci)
app.add_typer(code_formatters_app, name="lint")


def main() -> None:
    app()


if __name__ == "__main__":
    init_logging()
    main()
