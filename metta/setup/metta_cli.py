#!/usr/bin/env -S uv run
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import gitta as git
from metta.common.util.fs import get_repo_root
from metta.setup.components.base import SetupModuleStatus
from metta.setup.local_commands import app as local_app
from metta.setup.symlink_setup import app as symlink_app
from metta.setup.test_matrix import available_presets, run_preset
from metta.setup.tools.book import app as book_app
from metta.setup.utils import debug, error, info, success, warning
from metta.utils.live_run_monitor import app as run_monitor_app
from softmax.dashboard.report import app as softmax_system_health_app

if TYPE_CHECKING:
    from metta.setup.registry import SetupModule

VERSION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+(?:\.\d+)?)$")
DEFAULT_INITIAL_VERSION = "0.0.0.1"


class MettaCLI:
    def __init__(self):
        self.repo_root: Path = get_repo_root()
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
    callback=cli._init_all,
)


def _partition_supported_lint_files(paths: list[str]) -> tuple[list[str], list[str]]:
    python_files: list[str] = []
    cpp_files: list[str] = []
    seen: set[str] = set()

    for raw_path in paths:
        if not (raw_path and (path := raw_path.strip()) and path not in seen):
            continue
        seen.add(path)

        suffix = Path(path).suffix.lower()
        if path.endswith(".py"):
            python_files.append(path)
        elif suffix in {".cpp", ".hpp", ".h"}:
            cpp_files.append(path)

    return python_files, cpp_files


def _run_ruff(python_targets: list[str] | None, *, fix: bool) -> None:
    check_cmd = ["uv", "run", "--active", "ruff", "check"]
    format_cmd = ["uv", "run", "--active", "ruff", "format"]

    if fix:
        check_cmd.append("--fix")
    else:
        format_cmd.append("--check")

    for cmd in [format_cmd, check_cmd]:
        cmd.extend(python_targets or [])
        info(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=cli.repo_root, check=True)
        except subprocess.CalledProcessError as e:
            raise typer.Exit(e.returncode) from e


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
        cwd=cli.repo_root,
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


def _get_selected_modules(components: list[str] | None = None) -> list["SetupModule"]:
    from metta.setup.registry import get_all_modules

    return [
        m
        for m in get_all_modules()
        if (components is not None and m.name in components) or (components is None and m.is_enabled())
    ]


def _get_all_package_names() -> list[str]:
    """Return all valid package names in <repo_root>/packages/ (dirs with __init__.py or setup.py)."""
    repo_root = cli.repo_root
    packages_dir = repo_root / "packages"
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
        cmd_clean()

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

    if components:
        always_required_components = ["system", "core"]
        limited_components = always_required_components + [m for m in components if m not in always_required_components]
    else:
        limited_components = None
    modules = _get_selected_modules(limited_components)

    if not modules:
        info("No modules to install.")
        return

    info(f"\nInstalling {len(modules)} components...\n")

    for module in modules:
        info(f"[{module.name}] {module.description}")

        if module.install_once and module.check_installed() and not force:
            debug("  -> Already installed, skipping (use --force to reinstall)\n")
            continue

        try:
            module.install(non_interactive=non_interactive)
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
    import concurrent.futures

    modules = _get_selected_modules(components if components else None)
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
    from metta.setup.registry import get_all_modules

    modules = get_all_modules()
    module_map = {m.name: m for m in modules}

    if not (module := module_map.get(component)):
        error(f"Unknown component: {component}")
        info(f"Available components: {', '.join(sorted(module_map.keys()))}")
        raise typer.Exit(1)

    module.run(args or [])


@app.command(name="clean", help="Clean build artifacts and temporary files")
def cmd_clean(verbose: Annotated[bool, typer.Option("--verbose", help="Verbose output")] = False):
    def _remove_matching_dirs(base: Path, patterns: list[str], *, include_globs: bool = False) -> None:
        for pattern in patterns:
            candidates = base.glob(pattern) if include_globs else (base / pattern,)
            for path in candidates:
                if not path.exists() or not path.is_dir():
                    continue
                info(f"  Removing {path.relative_to(cli.repo_root)}...")
                subprocess.run(["chmod", "-R", "u+w", str(path)], cwd=cli.repo_root, check=False)
                subprocess.run(["rm", "-rf", str(path)], cwd=cli.repo_root, check=False)

    build_dir = cli.repo_root / "build"
    if build_dir.exists():
        info("  Removing root build directory...")
        shutil.rmtree(build_dir)

    mettagrid_dir = cli.repo_root / "packages" / "mettagrid"
    for build_name in ["build-debug", "build-release"]:
        build_path = mettagrid_dir / build_name
        if build_path.exists():
            info(f"  Removing packages/mettagrid/{build_name}...")
            shutil.rmtree(build_path)

    _remove_matching_dirs(cli.repo_root, ["bazel-*"], include_globs=True)
    _remove_matching_dirs(cli.repo_root, [".bazel_output"])
    if mettagrid_dir.exists():
        _remove_matching_dirs(mettagrid_dir, ["bazel-*"], include_globs=True)
        _remove_matching_dirs(mettagrid_dir, [".bazel_output"])

    cleanup_script = cli.repo_root / "devops" / "tools" / "cleanup_repo.py"
    if cleanup_script.exists():
        cmd = [str(cleanup_script)]
        if verbose:
            cmd.append("--verbose")
        try:
            subprocess.run(cmd, cwd=str(cli.repo_root), check=True)
        except subprocess.CalledProcessError as e:
            warning(f"  Cleanup script failed: {e}")


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
            subprocess.run([f"{cli.repo_root}/devops/git/push_child_repo.py", package, "-y"], check=True)
    except subprocess.CalledProcessError as exc:
        error(
            f"Failed to publish: {exc}. {tag_name} was still published to {remote}."
            + " Use --no-repo to skip pushing to github repo."
        )
        raise typer.Exit(exc.returncode) from exc

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


@app.command(name="lint", help="Run linting and formatting")
def cmd_lint(
    files: Annotated[Optional[list[str]], typer.Argument()] = None,
    fix: Annotated[bool, typer.Option("--fix", help="Apply fixes automatically")] = False,
    staged: Annotated[bool, typer.Option("--staged", help="Only lint staged files")] = False,
):
    python_targets: list[str] | None
    cpp_targets: list[str] | None

    if files is not None:
        python_targets, cpp_targets = _partition_supported_lint_files(files)
    elif staged:
        staged_output = git.run_git("diff", "--cached", "--name-only", "--diff-filter=ACM")
        staged_files = [f for f in staged_output.strip().split("\n") if f]
        python_targets, cpp_targets = _partition_supported_lint_files(staged_files)
    else:
        python_targets = None
        cpp_targets = None

    # Run ruff if the user specified targets that include python files or they did not specify any targets
    if python_targets is not None and not python_targets:
        info("No Python files to lint")
    else:
        _run_ruff(python_targets, fix=fix)

    # Run cpplint if the user specified targets that include c++ files or they did not specify any targets
    if cpp_targets is not None and not cpp_targets:
        info("No C++ files to lint")
    else:
        script_path = cli.repo_root / "packages" / "mettagrid" / "tests" / "cpplint.sh"
        res = subprocess.run(["bash", str(script_path)], cwd=cli.repo_root, check=False, capture_output=True)
        if res.returncode != 0:
            error("C++ linting failed")
            info(res.stderr.decode("utf-8"))
            raise typer.Exit(res.returncode)
        else:
            success("C++ linting passed!")


def _run_mettagrid_target(target: str, *, verbose: bool = False) -> int:
    mettagrid_dir = cli.repo_root / "packages" / "mettagrid"
    env = os.environ.copy()
    if verbose and "VERBOSE" not in env:
        env["VERBOSE"] = "1"
    cmd = ["make", target]
    return subprocess.run(cmd, cwd=mettagrid_dir, env=env, check=False).returncode


grid_app = typer.Typer(help="MettaGrid build and test helpers")
app.add_typer(grid_app, name="grid")


@grid_app.command("test", help="Run MettaGrid C++ unit tests")
def grid_test(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose build output.")] = False,
) -> None:
    info("Running MettaGrid C++ unit tests...")
    exit_code = _run_mettagrid_target("test", verbose=verbose)
    if exit_code != 0:
        error("MettaGrid C++ unit tests failed!")
        raise typer.Exit(exit_code)
    success("MettaGrid C++ unit tests passed!")


@grid_app.command("coverage", help="Run MettaGrid C++ tests with coverage enabled")
def grid_coverage(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose build output.")] = False,
) -> None:
    info("Running MettaGrid C++ coverage target...")
    exit_code = _run_mettagrid_target("coverage", verbose=verbose)
    if exit_code != 0:
        error("MettaGrid C++ coverage run failed!")
        raise typer.Exit(exit_code)
    success("MettaGrid C++ coverage run completed!")


@grid_app.command("benchmark", help="Build and run MettaGrid benchmarks")
def grid_benchmark(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose build output.")] = False,
) -> None:
    info("Running MettaGrid benchmarks...")
    exit_code = _run_mettagrid_target("benchmark", verbose=verbose)
    if exit_code != 0:
        error("MettaGrid benchmarks failed!")
        raise typer.Exit(exit_code)
    success("MettaGrid benchmarks completed!")


@app.command(name="ci", help="Run all Python unit tests and all MettaGrid C++ tests")
def cmd_ci(
    skip_app_backend: Annotated[
        bool,
        typer.Option(
            "--skip-app-backend",
            help="Skip the app_backend test suite regardless of environment settings.",
        ),
    ] = False,
) -> None:
    info("Running Python tests...")
    exit_code = run_preset(
        "ci",
        extra_pytest_args=(),
        skip_app_backend=True if skip_app_backend else None,
    )
    if exit_code != 0:
        error("Python tests failed!")
        raise typer.Exit(exit_code)
    success("Python tests passed!")

    info("\nBuilding and running C++ tests...")
    exit_code = _run_mettagrid_target("test", verbose=False)
    if exit_code != 0:
        error("C++ tests failed!")
        raise typer.Exit(exit_code)
    success("C++ tests passed!")
    success("\nAll CI tests passed!")


@app.command(name="benchmark", help="Run C++ and Python benchmarks for mettagrid")
def cmd_benchmark(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Verbose test output. Includes e.g. Python print statements within tests.",
        ),
    ] = False,
) -> None:
    info("Running mettagrid benchmarks...")
    info("Note: This may fail if Python environment is not properly configured.")
    info("If it fails, try running directly: cd packages/mettagrid && make benchmark")

    exit_code = _run_mettagrid_target("benchmark", verbose=verbose)
    if exit_code != 0:
        error("Benchmark execution failed!")
        info("\nTroubleshooting:")
        info("1. Try building first: cd packages/mettagrid && make build-prod")
        info("2. Run benchmark binary directly: ./build-release/test_mettagrid_env_benchmark")
        info("3. Run Python benchmarks: uv run pytest benchmarks/test_mettagrid_env_benchmark.py -v --benchmark-only")
        raise typer.Exit(exit_code)
    success("Benchmarks completed!")


@app.command(name="test", help="Run all Python unit tests", context_settings={"allow_extra_args": True})
def cmd_test(
    ctx: typer.Context,
    preset: Annotated[str, typer.Option("--preset", "-p", help="Test preset to run.")] = "default",
    skip_app_backend: Annotated[
        bool,
        typer.Option(
            "--skip-app-backend",
            help="Skip the app_backend suite (ci preset only).",
        ),
    ] = False,
) -> None:
    preset_normalized = preset.lower()
    if preset_normalized not in available_presets():
        error(f"Unknown preset '{preset}'. Available presets: {', '.join(available_presets())}")
        raise typer.Exit(1)

    extra_args = list(ctx.args or [])
    try:
        exit_code = run_preset(
            preset_normalized,
            extra_pytest_args=extra_args,
            skip_app_backend=True if skip_app_backend else None,
        )
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc

    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command(
    name="pytest",
    help="Run pytest with passed arguments",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def cmd_pytest(ctx: typer.Context):
    cmd = [
        "uv",
        "run",
        "pytest",
        "--benchmark-disable",
        "-n",
        "auto",
    ]
    if ctx.args:
        cmd.extend(ctx.args)
    try:
        subprocess.run(cmd, cwd=cli.repo_root, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode) from e


@app.command(name="tool", help="Run a tool from the tools/ directory", context_settings={"allow_extra_args": True})
def cmd_tool(
    tool_name: Annotated[str, typer.Argument(help="Name of the tool to run")],
    ctx: typer.Context,
):
    tool_path = cli.repo_root / "tools" / f"{tool_name}.py"
    if not tool_path.exists():
        error(f"Error: Tool '{tool_name}' not found at {tool_path}")
        raise typer.Exit(1)

    cmd = [str(tool_path)] + (ctx.args or [])
    try:
        subprocess.run(cmd, cwd=cli.repo_root, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode) from e


@app.command(name="shell", help="Start an IPython shell with Metta imports")
def cmd_shell():
    cmd = ["uv", "run", "--active", "metta/setup/shell.py"]
    try:
        subprocess.run(cmd, cwd=cli.repo_root, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode) from e


@app.command(name="go", help="Navigate to a Softmax Home shortcut", context_settings={"allow_extra_args": True})
def cmd_go(ctx: typer.Context):
    import webbrowser

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


# Report env details command
@app.command(name="report-env-details", help="Report environment details including UV project directory")
def cmd_report_env_details():
    """Report environment details."""
    import gitta

    info(f"UV Project Directory: {cli.repo_root}")
    info(f"Metta CLI Working Directory: {Path.cwd()}")
    if branch := gitta.get_current_branch():
        info(f"Git Branch: {branch}")
    if commit := gitta.get_current_commit():
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
    import sys

    # Find all arguments after 'clip' command
    clip_index = sys.argv.index("clip")
    args_after_clip = sys.argv[clip_index + 1 :]

    # Build command with codeclip and pass all arguments through
    cmd = ["codeclip"] + args_after_clip

    try:
        subprocess.run(cmd, cwd=cli.repo_root, check=False)
    except FileNotFoundError:
        error("Error: Command not found: codeclip")
        info("Run: metta install codebot")
        raise typer.Exit(1) from None


@app.command(name="gridworks", help="Start the Gridworks web UI", context_settings={"allow_extra_args": True})
def cmd_gridworks(ctx: typer.Context):
    cmd = ["./gridworks/start.py", *ctx.args]
    subprocess.run(cmd, cwd=cli.repo_root, check=False)


app.add_typer(run_monitor_app, name="run-monitor", help="Monitor training runs.")
app.add_typer(local_app, name="local")
app.add_typer(book_app, name="book")
app.add_typer(symlink_app, name="symlink-setup")
app.add_typer(softmax_system_health_app, name="softmax-system-health")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
