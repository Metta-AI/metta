#!/usr/bin/env -S uv run
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from metta.common.util.fs import get_repo_root
from metta.setup.local_commands import app as local_app
from metta.setup.symlink_setup import app as symlink_app
from metta.setup.tools.book import app as book_app
from metta.setup.utils import error, info, success, warning

app = typer.Typer(
    help="Metta Setup Tool - Configure and install development environment",
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

PYTHON_TEST_FOLDERS = [
    "tests",
    "mettascope/tests",
    "agent/tests",
    "app_backend/tests",
    "codebot/tests",
    "common/tests",
    "mettagrid/tests",
]


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


# Create a single CLI instance
cli = MettaCLI()


# Configure command
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
    cli._init_all()
    if component:
        configure_component(component)
    elif profile:
        from metta.setup.profiles import PROFILE_DEFINITIONS, UserType
        from metta.setup.saved_settings import get_saved_settings

        selected_user_type = UserType(profile)
        if selected_user_type in PROFILE_DEFINITIONS:
            saved_settings = get_saved_settings()
            saved_settings.apply_profile(selected_user_type)
            success(f"Configured as {selected_user_type.value} user.")
            info("\nRun 'metta install' to set up your environment.")
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


# Install command
@app.command(name="install", help="Install or update components")
def cmd_install(
    components: Annotated[Optional[list[str]], typer.Argument(help="Components to install")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force reinstall")] = False,
    no_clean: Annotated[bool, typer.Option("--no-clean", help="Skip cleaning before install")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive", help="Non-interactive mode")] = False,
):
    """Install or update components."""
    from metta.setup.registry import get_all_modules, get_enabled_setup_modules
    from metta.setup.saved_settings import get_saved_settings

    cli._init_all()

    if not get_saved_settings().exists():
        warning("No configuration found. Running setup wizard first...")
        cli.setup_wizard()

    if not no_clean:
        cmd_clean()

    if components:
        modules = get_all_modules()
    else:
        modules = get_enabled_setup_modules()

    if components:
        only_names = list(components)
        original_only = set(only_names)

        essential_modules = {"system", "core"}
        added_essentials = essential_modules - original_only

        for essential in essential_modules:
            if essential not in only_names:
                only_names.append(essential)

        if added_essentials:
            info(f"Note: Adding essential dependencies: {', '.join(sorted(added_essentials))}\n")

        modules = [m for m in modules if m.name in only_names]
        modules.sort(key=lambda m: (m.name not in essential_modules, m.name))

    if not modules:
        info("No modules to install.")
        return

    info(f"\nInstalling {len(modules)} components...\n")

    for module in modules:
        info(f"[{module.name}] {module.description}")

        if module.install_once and module.check_installed() and not force:
            info("  -> Already installed, skipping (use --force to reinstall)\n")
            continue

        try:
            module.install(non_interactive=non_interactive)
            print()
        except Exception as e:
            error(f"  Error: {e}\n")

    success("Installation complete!")


# Status command
@app.command(name="status", help="Show status of all components")
def cmd_status(
    components: Annotated[
        Optional[str], typer.Option("--components", help="Comma-separated list of components")
    ] = None,
    non_interactive: Annotated[bool, typer.Option("-n", "--non-interactive", help="Non-interactive mode")] = False,
):
    """Show status of all components."""
    import concurrent.futures

    from metta.setup.registry import get_all_modules

    cli._init_all()

    all_modules = get_all_modules()

    if components:
        requested_components = [c.strip() for c in components.split(",")]
        module_map = {m.name: m for m in all_modules}
        modules = []
        for comp in requested_components:
            if comp in module_map:
                modules.append(module_map[comp])
            else:
                warning(f"Unknown component: {comp}")
                info(f"Available components: {', '.join(sorted(module_map.keys()))}")
        if not modules:
            return
    else:
        modules = all_modules

    if not modules:
        warning("No modules found.")
        return

    applicable_modules = [m for m in modules if m.is_enabled()]
    if not applicable_modules:
        warning("No applicable modules found.")
        return

    module_status = {}

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
        installed = status_data["installed"]
        connected_as = status_data["connected_as"]
        expected = status_data["expected"]

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

    all_installed = all(module_status[name]["installed"] for name in module_status)
    all_connected = all(
        (module_status[name]["connected_as"] is not None or module_status[name]["expected"] is None)
        for name in module_status
        if module_status[name]["installed"]
    )

    if all_installed:
        if all_connected:
            success("All components are properly configured!")
        else:
            warning("Some components need authentication. Run 'metta install' to set them up.")
    else:
        warning("Some components are not installed. Run 'metta install' to set them up.")

    not_connected = [
        name
        for name, data in module_status.items()
        if data["installed"] and data["expected"] and data["connected_as"] is None
    ]

    if not_connected:
        console.print(f"\n[yellow]Components not connected: {', '.join(not_connected)}[/yellow]")
        console.print("This could be due to expired credentials, network issues, or broken installations.")

        if non_interactive:
            console.print(f"\nTo fix: metta install {' '.join(not_connected)} --force")
        elif sys.stdin.isatty():
            if typer.confirm("\nReinstall these components to fix connection issues?"):
                console.print(f"\nRunning: metta install {' '.join(not_connected)} --force")
                subprocess.run([sys.executable, __file__, "install"] + not_connected + ["--force"], cwd=cli.repo_root)

    not_installed = [name for name, data in module_status.items() if not data["installed"]]

    if not_installed:
        console.print(f"\n[yellow]Components not installed: {', '.join(not_installed)}[/yellow]")

        if non_interactive:
            console.print(f"\nTo fix: metta install {' '.join(not_installed)}")
        elif sys.stdin.isatty():
            if typer.confirm("\nInstall these components?"):
                console.print(f"\nRunning: metta install {' '.join(not_installed)}")
                subprocess.run([sys.executable, __file__, "install"] + not_installed, cwd=cli.repo_root)


# Run command
@app.command(name="run", help="Run component-specific commands")
def cmd_run(
    component: Annotated[str, typer.Argument(help="Component to run command for")],
    args: Annotated[Optional[list[str]], typer.Argument(help="Arguments to pass to the component")] = None,
):
    """Run component-specific commands."""
    from metta.setup.registry import get_all_modules

    cli._init_all()

    modules = get_all_modules()
    module_map = {m.name: m for m in modules}

    if not (module := module_map.get(component)):
        error(f"Unknown component: {component}")
        info(f"Available components: {', '.join(sorted(module_map.keys()))}")
        raise typer.Exit(1)

    module.run(args or [])


# Clean command
@app.command(name="clean", help="Clean build artifacts and temporary files")
def cmd_clean(verbose: Annotated[bool, typer.Option("--verbose", help="Verbose output")] = False):
    """Clean build artifacts and temporary files."""

    build_dir = cli.repo_root / "build"
    if build_dir.exists():
        info("  Removing root build directory...")
        shutil.rmtree(build_dir)

    mettagrid_dir = cli.repo_root / "mettagrid"
    for build_name in ["build-debug", "build-release"]:
        build_path = mettagrid_dir / build_name
        if build_path.exists():
            info(f"  Removing mettagrid/{build_name}...")
            shutil.rmtree(build_path)

    cleanup_script = cli.repo_root / "devops" / "tools" / "cleanup_repo.py"
    if cleanup_script.exists():
        cmd = [str(cleanup_script)]
        if verbose:
            cmd.append("--verbose")
        try:
            subprocess.run(cmd, cwd=str(cli.repo_root), check=True)
        except subprocess.CalledProcessError as e:
            warning(f"  Cleanup script failed: {e}")


# Lint command
@app.command(name="lint", help="Run linting and formatting")
def cmd_lint(
    fix: Annotated[bool, typer.Option("--fix", help="Apply fixes automatically")] = False,
    staged: Annotated[bool, typer.Option("--staged", help="Only lint staged files")] = False,
):
    """Run linting and formatting."""
    files = []
    if staged:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=cli.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f]
        if not files:
            return

    check_cmd = ["uv", "run", "--active", "ruff", "check"]
    format_cmd = ["uv", "run", "--active", "ruff", "format"]
    cmds = [format_cmd, check_cmd]

    if fix:
        check_cmd.append("--fix")
    else:
        format_cmd.append("--check")

    if files:
        for cmd in cmds:
            cmd.extend(files)

    for cmd in cmds:
        try:
            info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=cli.repo_root, check=True)
        except subprocess.CalledProcessError as e:
            raise typer.Exit(e.returncode) from e


# CI command
@app.command(name="ci", help="Run all Python unit tests and all Mettagrid C++ tests")
def cmd_ci():
    """Run all Python unit tests and all Mettagrid C++ tests."""

    cli._init_all()

    info("Running Python tests...")
    python_test_cmd = [
        "uv",
        "run",
        "pytest",
        *PYTHON_TEST_FOLDERS,
        "--benchmark-disable",
        "-n",
        "auto",
    ]

    try:
        subprocess.run(python_test_cmd, cwd=cli.repo_root, check=True)
        success("Python tests passed!")
    except subprocess.CalledProcessError as e:
        error("Python tests failed!")
        raise typer.Exit(e.returncode) from e

    info("\nBuilding and running C++ tests...")
    mettagrid_dir = cli.repo_root / "mettagrid"

    try:
        subprocess.run(["cmake", "--preset", "benchmark"], cwd=mettagrid_dir, check=True)
        subprocess.run(["cmake", "--build", "build-release"], cwd=mettagrid_dir, check=True)
        build_dir = mettagrid_dir / "build-release"
        subprocess.run(["ctest", "-L", "benchmark", "--output-on-failure"], cwd=build_dir, check=True)
        success("C++ tests passed!")
    except subprocess.CalledProcessError as e:
        error("C++ tests failed!")
        raise typer.Exit(e.returncode) from e

    success("\nAll CI tests passed!")


# Test command
@app.command(name="test", help="Run all Python unit tests", context_settings={"allow_extra_args": True})
def cmd_test(ctx: typer.Context):
    """Run all Python unit tests."""
    cmd = [
        "uv",
        "run",
        "pytest",
        *PYTHON_TEST_FOLDERS,
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


# Pytest command
@app.command(
    name="pytest",
    help="Run pytest with passed arguments",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def cmd_pytest(ctx: typer.Context):
    """Run pytest with custom arguments."""
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


# Tool command
@app.command(name="tool", help="Run a tool from the tools/ directory", context_settings={"allow_extra_args": True})
def cmd_tool(
    tool_name: Annotated[str, typer.Argument(help="Name of the tool to run")],
    ctx: typer.Context,
):
    """Run a tool from the tools/ directory."""
    tool_path = cli.repo_root / "tools" / f"{tool_name}.py"
    if not tool_path.exists():
        error(f"Error: Tool '{tool_name}' not found at {tool_path}")
        raise typer.Exit(1)

    cmd = [str(tool_path)] + (ctx.args or [])
    try:
        subprocess.run(cmd, cwd=cli.repo_root, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode) from e


# Shell command
@app.command(name="shell", help="Start an IPython shell with Metta imports")
def cmd_shell():
    """Start IPython shell."""
    cmd = ["uv", "run", "--active", "metta/setup/shell.py"]
    try:
        subprocess.run(cmd, cwd=cli.repo_root, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode) from e


# Go command
@app.command(name="go", help="Navigate to a Softmax Home shortcut", context_settings={"allow_extra_args": True})
def cmd_go(ctx: typer.Context):
    """Navigate to Softmax Home shortcut."""
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


# Clip command
@app.command(name="clip", help="Copy subsets of codebase for LLM contexts", context_settings={"allow_extra_args": True})
def cmd_clip(ctx: typer.Context):
    """Run codeclip tool."""
    cmd = ["codeclip"]
    if ctx.args:
        cmd.extend(ctx.args)
    try:
        subprocess.run(cmd, cwd=cli.repo_root, check=False)
    except FileNotFoundError:
        error("Error: Command not found: codeclip")
        info("Run: metta install codebot")
        raise typer.Exit(1) from None


app.add_typer(local_app, name="local")
app.add_typer(book_app, name="book")
app.add_typer(symlink_app, name="symlink-setup")


@app.callback()
def main_callback():
    """Handle initialization checks."""
    pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
