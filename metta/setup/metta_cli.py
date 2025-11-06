#!/usr/bin/env -S uv run
from __future__ import annotations

import concurrent.futures
import pathlib
import re
import shutil
import subprocess
import sys
import typing
import webbrowser

import rich.console
import rich.progress
import rich.table
import typer

import gitta as git
import metta.common.util.fs
import metta.setup._path_setup  # noqa: F401
import metta.setup.components.base
import metta.setup.local_commands
import metta.setup.profiles
import metta.setup.saved_settings
import metta.setup.symlink_setup
import metta.setup.tools.book
import metta.setup.tools.ci_runner
import metta.setup.tools.code_formatters
import metta.setup.tools.test_runner.test_cpp
import metta.setup.tools.test_runner.test_python
import metta.setup.utils
import metta.tools.utils.auto_config
import metta.utils.live_run_monitor
import softmax.dashboard.report

if typing.TYPE_CHECKING:
    import metta.setup.registry

VERSION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+(?:\.\d+)?)$")
DEFAULT_INITIAL_VERSION = "0.0.0.1"


class MettaCLI:
    def __init__(self):
        self.repo_root: pathlib.Path = metta.common.util.fs.get_repo_root()
        self._components_initialized = False

    def _init_all(self):
        """Initialize all components - used by commands that need everything."""
        if self._components_initialized:
            return

        metta.setup.utils.import_all_modules_from_subpackage("metta.setup", "components")
        self._components_initialized = True

    def setup_wizard(self, non_interactive: bool = False):
        metta.setup.utils.header("Welcome to Metta!\n\n")
        metta.setup.utils.info(
            "Note: You can run 'metta configure <component>' to change component-level settings later.\n"
        )

        saved_settings = metta.setup.saved_settings.get_saved_settings()
        if saved_settings.exists():
            metta.setup.utils.info("Current configuration:")
            metta.setup.utils.info(f"Profile: {saved_settings.user_type.value}")
            metta.setup.utils.info(f"Mode: {'custom' if saved_settings.is_custom_config else 'profile'}")
            metta.setup.utils.info("\nEnabled components:")
            components = saved_settings.get_components()
            for comp, settings in components.items():
                if settings.get("enabled"):
                    metta.setup.utils.success(f"  + {comp}")
            metta.setup.utils.info("\n")

        choices = [(ut, ut.get_description()) for ut in metta.setup.profiles.UserType]

        current_user_type = saved_settings.user_type if saved_settings.exists() else None

        result = metta.setup.utils.prompt_choice(
            "Select configuration:",
            choices,
            current=current_user_type,
            non_interactive=non_interactive,
        )

        if result == metta.setup.profiles.UserType.CUSTOM:
            self._custom_setup(non_interactive=non_interactive)
        else:
            saved_settings.apply_profile(result)
            metta.setup.utils.success(f"\nConfigured as {result.value} user.")
        metta.setup.utils.info("\nRun 'metta install' to set up your environment.")

    def _custom_setup(self, non_interactive: bool = False):
        user_type = metta.setup.utils.prompt_choice(
            "Select base profile for custom configuration:",
            [
                (ut, ut.get_description())
                for ut in metta.setup.profiles.UserType
                if ut != metta.setup.profiles.UserType.CUSTOM
            ],
            default=metta.setup.profiles.UserType.EXTERNAL,
            non_interactive=non_interactive,
        )

        saved_settings = metta.setup.saved_settings.get_saved_settings()
        saved_settings.setup_custom_profile(user_type)

        metta.setup.utils.info("\nCustomize components:")
        all_modules = metta.setup.registry.get_all_modules()

        for module in all_modules:
            current_enabled = saved_settings.is_component_enabled(module.name)

            enabled = metta.setup.utils.prompt_choice(
                f"Enable {module.name} ({module.description})?",
                [(True, "Yes"), (False, "No")],
                default=current_enabled,
                current=current_enabled,
                non_interactive=non_interactive,
            )

            profile_default = (
                metta.setup.profiles.PROFILE_DEFINITIONS.get(user_type, {})
                .get("components", {})
                .get(module.name, {})
                .get("enabled", False)
            )
            if enabled != profile_default:
                saved_settings.set(f"components.{module.name}.enabled", enabled)

        metta.setup.utils.success("\nCustom configuration saved.")
        metta.setup.utils.info("\nRun 'metta install' to set up your environment.")

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


def _bump_version(version: str) -> str:
    parts = version.split(".")
    bumped = parts[:-1] + [str(int(parts[-1]) + 1)]
    return ".".join(bumped)


def _validate_version_format(version: str) -> None:
    if not VERSION_PATTERN.match(version):
        metta.setup.utils.error(f"Invalid version '{version}'. Expected numeric segments like '1.2.3' or '1.2.3.4'.")
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
        metta.setup.utils.error(f"Tag '{tag_name}' already exists.")
        raise typer.Exit(1)


@app.command(name="configure", help="Configure Metta settings")
def cmd_configure(
    component: typing.Annotated[typing.Optional[str], typer.Argument(help="Specific component to configure")] = None,
    profile: typing.Annotated[
        typing.Optional[str],
        typer.Option(
            "--profile",
            help="Set user profile",
        ),
    ] = None,
    non_interactive: typing.Annotated[bool, typer.Option("--non-interactive", help="Non-interactive mode")] = False,
):
    """Configure Metta settings."""
    if component:
        if profile:
            metta.setup.utils.error("Cannot configure a component and a profile at the same time.")
            raise typer.Exit(1)
        configure_component(component)
    elif profile:
        selected_user_type = metta.setup.profiles.UserType(profile)
        if selected_user_type in metta.setup.profiles.PROFILE_DEFINITIONS:
            saved_settings = metta.setup.saved_settings.get_saved_settings()
            saved_settings.apply_profile(selected_user_type)
            metta.setup.utils.success(f"Configured as {selected_user_type.value} user.")
        else:
            metta.setup.utils.error(f"Unknown profile: {profile}")
            raise typer.Exit(1)
    else:
        cli.setup_wizard(non_interactive=non_interactive)


def configure_component(component_name: str):
    modules = metta.setup.registry.get_all_modules()
    module_map = {m.name: m for m in modules}

    if not (module := module_map.get(component_name)):
        metta.setup.utils.error(f"Unknown component: {component_name}")
        metta.setup.utils.info(f"Available components: {', '.join(sorted(module_map.keys()))}")
        raise typer.Exit(1)

    options = module.get_configuration_options()
    if not options:
        metta.setup.utils.info(f"Component '{component_name}' has no configuration options.")
        return
    module.configure()


def _get_selected_modules(
    components: list[str] | None = None,
) -> list[metta.setup.components.base.SetupModule]:
    return [
        m
        for m in metta.setup.registry.get_all_modules()
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
    components: typing.Annotated[typing.Optional[list[str]], typer.Argument(help="Components to install")] = None,
    profile: typing.Annotated[
        typing.Optional[str], typer.Option("--profile", help="Profile to configure before installing")
    ] = None,
    force: typing.Annotated[bool, typer.Option("--force", help="Force reinstall")] = False,
    no_clean: typing.Annotated[bool, typer.Option("--no-clean", help="Skip cleaning before install")] = False,
    non_interactive: typing.Annotated[bool, typer.Option("--non-interactive", help="Non-interactive mode")] = False,
    check_status: typing.Annotated[bool, typer.Option("--check-status", help="Check status after installation")] = True,
):
    if not no_clean:
        cmd_clean(force=force)

    # A profile must exist before installing. If installing in non-interactive mode,
    # the target profile must be specified with --profile. If in interactive mode and
    # no profile is specified, the setup wizard will be run.
    profile_exists = metta.setup.saved_settings.get_saved_settings().exists()
    if non_interactive and not profile_exists and not profile:
        metta.setup.utils.error("Must specify a profile if installing in non-interactive mode without an existing one.")
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
        metta.setup.utils.info("No modules to install.")
        return

    metta.setup.utils.info(f"\nInstalling {len(modules)} components...\n")

    for module in modules:
        metta.setup.utils.info(f"[{module.name}] {module.description}")

        if module.install_once and module.check_installed() and not force:
            metta.setup.utils.debug("  -> Already installed, skipping (use --force to reinstall)\n")
            continue

        try:
            module.install(non_interactive=non_interactive, force=force)
            print()
        except Exception as e:
            metta.setup.utils.error(f"  Error: {e}\n")

    if not non_interactive and check_status:
        cmd_status(components=components, non_interactive=non_interactive)


@app.command(name="status", help="Show status of components")
def cmd_status(
    components: typing.Annotated[
        typing.Optional[list[str]],
        typer.Option("--components", help="Comma-separated list of components. Defaults to all enabled components."),
    ] = None,
    non_interactive: typing.Annotated[
        bool, typer.Option("-n", "--non-interactive", help="Non-interactive mode")
    ] = False,
):
    modules = _get_selected_modules(components if components else None)
    if not modules:
        metta.setup.utils.warning("No modules to check.")
        return

    modules_by_name = {m.name: m for m in modules}
    module_status: dict[str, metta.setup.components.base.SetupModuleStatus] = {}

    console = rich.console.Console()
    with rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn("[progress.description]{task.description}"),
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

    table = rich.table.Table(title="Component Status", show_header=True, header_style="bold magenta")
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

    console = rich.console.Console()
    console.print(table)

    policy_decision = metta.tools.utils.auto_config.auto_policy_storage_decision()
    if policy_decision.using_remote and policy_decision.base_prefix:
        if policy_decision.reason == "env_override":
            metta.setup.utils.success(
                f"Policy storage: S3 uploads enabled via POLICY_REMOTE_PREFIX → {policy_decision.base_prefix}/<run>."
            )
        else:
            metta.setup.utils.success(
                f"Policy storage: Softmax S3 uploads active → {policy_decision.base_prefix}/<run>."
            )
    elif policy_decision.reason == "not_connected" and policy_decision.base_prefix:
        metta.setup.utils.warning(
            "Policy storage: local only. Run 'aws sso login --profile softmax' to enable uploads to "
            f"{policy_decision.base_prefix}/<run>."
        )
    elif policy_decision.reason == "aws_not_enabled":
        metta.setup.utils.info("Policy storage: local only (AWS component disabled).")
    elif policy_decision.reason == "no_base_prefix":
        metta.setup.utils.info(
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
    component: typing.Annotated[str, typer.Argument(help="Component to run command for")],
    args: typing.Annotated[
        typing.Optional[list[str]], typer.Argument(help="Arguments to pass to the component")
    ] = None,
):
    modules = metta.setup.registry.get_all_modules()
    module_map = {m.name: m for m in modules}

    if not (module := module_map.get(component)):
        metta.setup.utils.error(f"Unknown component: {component}")
        metta.setup.utils.info(f"Available components: {', '.join(sorted(module_map.keys()))}")
        raise typer.Exit(1)

    module.run(args or [])


@app.command(name="clean", help="Clean build artifacts and temporary files")
def cmd_clean(
    verbose: typing.Annotated[bool, typer.Option("--verbose", help="Verbose output")] = False,
    force: typing.Annotated[bool, typer.Option("--force", help="Force clean")] = False,
):
    def _remove_matching_dirs(base: pathlib.Path, patterns: list[str], *, include_globs: bool = False) -> None:
        for pattern in patterns:
            candidates = base.glob(pattern) if include_globs else (base / pattern,)
            for path in candidates:
                if not path.exists() or not path.is_dir():
                    continue
                metta.setup.utils.info(f"  Removing {path.relative_to(cli.repo_root)}...")
                subprocess.run(["chmod", "-R", "u+w", str(path)], cwd=cli.repo_root, check=False)
                subprocess.run(["rm", "-rf", str(path)], cwd=cli.repo_root, check=False)

    build_dir = cli.repo_root / "build"
    if build_dir.exists():
        metta.setup.utils.info("  Removing root build directory...")
        shutil.rmtree(build_dir)

    mettagrid_dir = cli.repo_root / "packages" / "mettagrid"
    for build_name in ["build-debug", "build-release"]:
        build_path = mettagrid_dir / build_name
        if build_path.exists():
            metta.setup.utils.info(f"  Removing packages/mettagrid/{build_name}...")
            shutil.rmtree(build_path)

    _remove_matching_dirs(cli.repo_root, ["bazel-*"], include_globs=True)
    _remove_matching_dirs(cli.repo_root, [".bazel_output"])
    if mettagrid_dir.exists():
        _remove_matching_dirs(mettagrid_dir, ["bazel-*"], include_globs=True)
        _remove_matching_dirs(mettagrid_dir, [".bazel_output"])

    nim_generated_dir = cli.repo_root / "packages" / "mettagrid" / "nim" / "mettascope" / "bindings" / "generated"
    if force and nim_generated_dir.exists():
        shutil.rmtree(nim_generated_dir)

    cleanup_script = cli.repo_root / "devops" / "tools" / "cleanup_repo.py"
    if cleanup_script.exists():
        cmd = [str(cleanup_script)]
        if verbose:
            cmd.append("--verbose")
        try:
            subprocess.run(cmd, cwd=str(cli.repo_root), check=True)
        except subprocess.CalledProcessError as e:
            metta.setup.utils.warning(f"  Cleanup script failed: {e}")


@app.command(name="publish", help="Create and push a release tag for a package")
def cmd_publish(
    package: typing.Annotated[str, typer.Argument(help="Package to publish")],
    version_override: typing.Annotated[
        typing.Optional[str],
        typer.Option("--version", "-v", help="Explicit version to tag (digits separated by dots)"),
    ] = None,
    dry_run: typing.Annotated[bool, typer.Option("--dry-run", help="Preview actions without tagging")] = False,
    no_repo: typing.Annotated[bool, typer.Option("--no-repo", help="Don't push the github repo")] = False,
    remote: typing.Annotated[str, typer.Option("--remote", help="Git remote to push the tag to")] = "origin",
    force: typing.Annotated[bool, typer.Option("--force", help="Bypass branch and clean checks")] = False,
):
    package = package.lower()
    if package not in _get_all_package_names():
        metta.setup.utils.error(
            f"Unsupported package '{package}'. Supported packages: {', '.join(sorted(_get_all_package_names()))}."
        )
        raise typer.Exit(1)

    prefix = f"{package}-v"
    try:
        metta.setup.utils.info(f"Fetching tags from {remote}...")
        git.run_git("fetch", remote, "--tags", "--force")
    except subprocess.CalledProcessError as exc:
        metta.setup.utils.error(f"Failed to fetch tags from {remote}: {exc}")
        raise typer.Exit(exc.returncode) from exc

    try:
        status_output = git.run_git("status", "--porcelain")
    except subprocess.CalledProcessError as exc:
        metta.setup.utils.error(f"Failed to read git status: {exc}")
        raise typer.Exit(exc.returncode) from exc

    if status_output.strip() and not force:
        metta.setup.utils.error(
            "Working tree is not clean. Commit, stash, or clean changes before publishing (use --force to override)."
        )
        raise typer.Exit(1)

    try:
        current_branch = git.run_git("rev-parse", "--abbrev-ref", "HEAD")
        current_commit = git.run_git("rev-parse", "HEAD")
    except subprocess.CalledProcessError as exc:
        metta.setup.utils.error(f"Failed to determine git state: {exc}")
        raise typer.Exit(exc.returncode) from exc

    if current_branch not in {"main"} and not force:
        metta.setup.utils.error(
            "Publishing is only supported from the main branch. Switch to 'main' or pass --force to override."
        )
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

    metta.setup.utils.info("Release summary:\n")
    metta.setup.utils.info(f"  Package: {package}")
    metta.setup.utils.info(f"  Current branch: {current_branch}")
    metta.setup.utils.info(f"  Commit: {current_commit}")
    metta.setup.utils.info(f"  Tag: {tag_name}")
    if latest_tag:
        metta.setup.utils.info(f"  Previous tag: {latest_tag}")
    else:
        metta.setup.utils.info("  Previous tag: none")
    if force:
        metta.setup.utils.warning("Force mode enabled: branch and clean checks were bypassed.")
    metta.setup.utils.info("")

    publish_mettagrid_after = False

    if dry_run:
        metta.setup.utils.success("Dry run: no tag created. Run without --dry-run to proceed.")
        return

    if package == "cogames":
        publish_mettagrid_after = typer.confirm(
            "Cogames depends on mettagrid. Publish mettagrid after this tag?",
            default=True,
        )

    if not typer.confirm("Create and push this tag?", default=True):
        metta.setup.utils.info("Publishing aborted.")
        return

    try:
        metta.setup.utils.info(f"Tagging {package} {target_version}...")
        git.run_git("tag", "-a", tag_name, "-m", f"Release {package} {target_version}")
        git.run_git("push", remote, tag_name)
    except subprocess.CalledProcessError as exc:
        metta.setup.utils.error(f"Failed to tag: {exc}.")
        raise typer.Exit(exc.returncode) from exc

    metta.setup.utils.success(f"Published {tag_name} to {remote}.")

    try:
        if not no_repo:
            metta.setup.utils.info(f"Pushing {package} as child repo...")
            subprocess.run([f"{cli.repo_root}/devops/git/push_child_repo.py", package, "-y"], check=True)
    except subprocess.CalledProcessError as exc:
        metta.setup.utils.error(
            f"Failed to publish: {exc}. {tag_name} was still published to {remote}."
            + " Use --no-repo to skip pushing to github repo."
        )
        raise typer.Exit(exc.returncode) from exc

    if publish_mettagrid_after:
        metta.setup.utils.info("")
        metta.setup.utils.info("Starting mettagrid publish flow...")
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
    files: typing.Annotated[typing.Optional[list[str]], typer.Argument()] = None,
    fix: typing.Annotated[bool, typer.Option("--fix", help="Apply fixes automatically")] = False,
    staged: typing.Annotated[bool, typer.Option("--staged", help="Only lint staged files")] = False,
    check: typing.Annotated[bool, typer.Option("--check", help="Check formatting without modifying files")] = False,
    type: typing.Annotated[
        typing.Optional[str],
        typer.Option(
            "--type",
            "-t",
            help="Comma-separated file types (e.g., 'json,yaml'). Default: all detected types.",
        ),
    ] = None,
):
    """Run linting and formatting on code files.

    By default, formats and lints all detected file types. Use --type to restrict to specific types.

    Examples:
        metta lint                    # Format and lint all detected files
        metta lint --fix              # Format and lint with auto-fix
        metta lint --type json,yaml   # Only format JSON and YAML files
        metta lint --check            # Check formatting without modifying
        metta lint --staged --fix     # Format and lint only staged files
    """
    # Get available formatters
    formatters = metta.setup.tools.code_formatters.get_formatters(cli.repo_root)

    # Determine which files to process
    if files is not None:
        target_files = files
    elif staged:
        staged_output = git.run_git("diff", "--cached", "--name-only", "--diff-filter=ACM")
        target_files = [f for f in staged_output.strip().split("\n") if f]
    else:
        target_files = None

    # Partition files by type
    if target_files is not None:
        files_by_type = metta.setup.tools.code_formatters.partition_files_by_type(target_files)
    else:
        # No specific files provided - will format all files of each type
        files_by_type = {}

    # Determine which types to format
    if type:
        try:
            types_to_format = metta.setup.tools.code_formatters.parse_format_types(type, formatters)
        except ValueError as e:
            metta.setup.utils.error(str(e))
            raise typer.Exit(1) from e
    else:
        # Default: format all detected types (or all types if no files specified)
        if files_by_type:
            types_to_format = list(files_by_type.keys())
        elif target_files is not None:
            # Files were specified but none have supported extensions
            metta.setup.utils.info("No files with supported extensions found")
            return
        else:
            # No specific files - format all supported types
            types_to_format = ["python", "json", "markdown", "shell", "toml", "yaml"]
            if "cpp" in formatters:
                types_to_format.append("cpp")

    failed_formatters = []
    failed_linters = []

    # Run formatters for each type
    for file_type in types_to_format:
        if file_type not in formatters:
            continue

        formatter = formatters[file_type]
        type_files = files_by_type.get(file_type) if files_by_type else None

        # Skip if we have a file list and no files of this type
        if files_by_type and not type_files:
            continue

        # Run formatter
        check_mode = check or not fix
        success_fmt = metta.setup.tools.code_formatters.run_formatter(
            file_type,
            formatter,
            cli.repo_root,
            check_only=check_mode,
            files=type_files,
        )

        # Only treat as failure if formatter ran and failed
        # If check_mode is True and formatter doesn't support check, it returns False but that's not a failure
        if not success_fmt:
            # If we're in check mode and the formatter doesn't have a check_cmd, ignore the failure
            if check_mode and formatter.check_cmd is None:
                # This is expected - formatter doesn't support check mode, was skipped
                pass
            else:
                # This is an actual failure
                failed_formatters.append(formatter.name)

    # Run Python linting (ruff check) if Python files are involved
    if "python" in types_to_format:
        python_files = files_by_type.get("python") if files_by_type else None

        if python_files is not None and not python_files:
            metta.setup.utils.info("No Python files to lint")
        else:
            check_cmd = ["uv", "run", "--active", "ruff", "check"]
            if fix:
                check_cmd.append("--fix")
            if python_files:
                check_cmd.extend(python_files)

            metta.setup.utils.info(f"Running: {' '.join(check_cmd)}")
            try:
                subprocess.run(check_cmd, cwd=cli.repo_root, check=True)
            except subprocess.CalledProcessError:
                failed_linters.append("Python (ruff check)")

    # Run C++ linting if C++ files are involved
    if "cpp" in types_to_format and "cpp" in formatters:
        cpp_files = files_by_type.get("cpp") if files_by_type else None

        if cpp_files is not None and not cpp_files:
            metta.setup.utils.info("No C++ files to lint")
        else:
            script_path = cli.repo_root / "packages" / "mettagrid" / "tests" / "cpplint.sh"
            res = subprocess.run(["bash", str(script_path)], cwd=cli.repo_root, check=False, capture_output=True)
            if res.returncode != 0:
                failed_linters.append("C++")
                metta.setup.utils.error("C++ linting failed")
                metta.setup.utils.info(res.stderr.decode("utf-8"))

    # Print summary
    if failed_formatters or failed_linters:
        if failed_formatters:
            metta.setup.utils.error(f"Formatting failed for: {', '.join(failed_formatters)}")
        if failed_linters:
            metta.setup.utils.error(f"Linting failed for: {', '.join(failed_linters)}")
        raise typer.Exit(1)
    else:
        metta.setup.utils.success("All linting and formatting complete")


@app.command(name="tool", help="Run a tool from the tools/ directory", context_settings={"allow_extra_args": True})
def cmd_tool(
    tool_name: typing.Annotated[str, typer.Argument(help="Name of the tool to run")],
    ctx: typer.Context,
):
    tool_path = cli.repo_root / "tools" / f"{tool_name}.py"
    if not tool_path.exists():
        metta.setup.utils.error(f"Error: Tool '{tool_name}' not found at {tool_path}")
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
    if not ctx.args:
        metta.setup.utils.error("Please specify a shortcut (e.g., 'metta go g' for GitHub)")
        metta.setup.utils.info("\nCommon shortcuts:")
        metta.setup.utils.info("  g    - GitHub")
        metta.setup.utils.info("  w    - Weights & Biases")
        metta.setup.utils.info("  o    - Observatory")
        metta.setup.utils.info("  d    - Datadog")
        return

    shortcut = ctx.args[0]
    url = f"https://home.softmax-research.net/{shortcut}"

    metta.setup.utils.info(f"Opening {url}...")
    webbrowser.open(url)


# Report env details command
@app.command(name="report-env-details", help="Report environment details including UV project directory")
def cmd_report_env_details():
    """Report environment details."""
    metta.setup.utils.info(f"UV Project Directory: {cli.repo_root}")
    metta.setup.utils.info(f"Metta CLI Working Directory: {pathlib.Path.cwd()}")
    if branch := git.get_current_branch():
        metta.setup.utils.info(f"Git Branch: {branch}")
    if commit := git.get_current_commit():
        metta.setup.utils.info(f"Git Commit: {commit}")


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
        subprocess.run(cmd, cwd=cli.repo_root, check=False)
    except FileNotFoundError:
        metta.setup.utils.error("Error: Command not found: codeclip")
        metta.setup.utils.info("Run: metta install codebot")
        raise typer.Exit(1) from None


@app.command(name="gridworks", help="Start the Gridworks web UI", context_settings={"allow_extra_args": True})
def cmd_gridworks(ctx: typer.Context):
    cmd = ["./gridworks/start.py", *ctx.args]
    subprocess.run(cmd, cwd=cli.repo_root, check=False)


app.add_typer(metta.utils.live_run_monitor.app, name="run-monitor", help="Monitor training runs.")
app.add_typer(metta.setup.local_commands.app, name="local")
app.add_typer(metta.setup.tools.book.app, name="book")
app.add_typer(metta.setup.symlink_setup.app, name="symlink-setup")
app.add_typer(softmax.dashboard.report.app, name="softmax-system-health")
app.add_typer(metta.setup.tools.test_runner.test_python.app, name="pytest")
app.add_typer(metta.setup.tools.test_runner.test_cpp.app, name="cpptest")
app.command(
    name="ci",
    help="Run CI checks locally",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True, "allow_interspersed_args": False},
)(metta.setup.tools.ci_runner.cmd_ci)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
