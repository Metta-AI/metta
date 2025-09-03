#!/usr/bin/env -S uv run
import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

# Minimal imports needed for all commands (or safe minimal imports tested for non-slowness)
from metta.common.util.fs import get_repo_root
from metta.setup.profiles import PROFILE_DEFINITIONS, UserType
from metta.setup.saved_settings import get_saved_settings
from metta.setup.utils import error, header, import_all_modules_from_subpackage, info, prompt_choice, success

# Type hints only
if TYPE_CHECKING:
    from metta.setup.local_commands import LocalCommands
    from metta.setup.symlink_setup import PathSetup
    from metta.setup.tools.book import BookCommands

# Shared list of test folders for Python tests
PYTHON_TEST_FOLDERS = [
    "tests",
    "mettascope/tests",
    "agent/tests",
    "app_backend/tests",
    "codebot/tests",
    "common/tests",
    "mettagrid/tests",
]


@dataclass
class CommandConfig:
    """Configuration for a single command."""

    help: str  # Help text shown in 'metta --help'
    handler: Optional[str] = None  # Method name in MettaCLI
    needs_config: bool = False
    needs_components: bool = False
    pass_unknown_args: bool = False
    subprocess_cmd: Optional[List[str]] = None  # For simple subprocess commands
    add_help: bool = True  # Whether subcommand accepts --help
    parser_setup: Optional[Callable[[argparse.ArgumentParser], None]] = None  # Parser setup function


# Parser setup functions for commands
def _setup_configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("component", nargs="?", help="Specific component to configure. If omitted, runs setup wizard.")
    parser.add_argument("--non-interactive", action="store_true", help="Non-interactive mode")
    # Profile choices will be added dynamically in _build_parser


def _setup_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("component", help="Component to run command for")
    parser.add_argument("args", nargs="*", help="Arguments to pass to the component")


def _setup_install_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("components", nargs="*", help="Components to install")
    parser.add_argument("--force", action="store_true", help="Force reinstall")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning before install")
    parser.add_argument("--non-interactive", action="store_true", help="Non-interactive mode")


def _setup_status_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--components", help="Comma-separated list of components")
    parser.add_argument("-n", "--non-interactive", action="store_true", help="Non-interactive mode")


def _setup_symlink_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="Replace existing metta command")


def _setup_lint_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fix", action="store_true", help="Apply fixes automatically")
    parser.add_argument("--staged", action="store_true", help="Only lint staged files")


def _setup_tool_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tool_name", help="Name of the tool to run")


# Command registry with all command definitions
COMMAND_REGISTRY: Dict[str, CommandConfig] = {
    # Simple subprocess commands
    "clip": CommandConfig(
        help="Copy subsets of codebase for LLM contexts",
        subprocess_cmd=["codeclip"],
        pass_unknown_args=True,
        add_help=False,  # codeclip handles its own --help
    ),
    "book": CommandConfig(
        help="Interactive marimo notebook commands",
        handler="cmd_book",
        needs_config=True,
        pass_unknown_args=True,
        add_help=False,  # Let BookCommands handle its own help
    ),
    "pytest": CommandConfig(
        help="Run pytest with passed arguments",
        subprocess_cmd=[
            "uv",
            "run",
            "pytest",
            "--benchmark-disable",
            "-n",
            "auto",
        ],
        pass_unknown_args=True,
    ),
    "test": CommandConfig(
        help="Run all Python unit tests",
        subprocess_cmd=[
            "uv",
            "run",
            "pytest",
            *PYTHON_TEST_FOLDERS,
            "--benchmark-disable",
            "-n",
            "auto",
        ],
    ),
    "ci": CommandConfig(
        help="Run all Python unit tests and all Mettagrid C++ tests",
        handler="cmd_ci",
        needs_config=True,  # Needs repo_root
    ),
    "tool": CommandConfig(
        help="Run a tool from the tools/ directory",
        handler="cmd_tool",
        pass_unknown_args=True,
        parser_setup=_setup_tool_parser,
    ),
    "shell": CommandConfig(
        help="Start an IPython shell with Metta imports",
        subprocess_cmd=["uv", "run", "--active", "metta/setup/shell.py"],
        needs_config=True,  # Needs repo_root
    ),
    "report-env-details": CommandConfig(
        help="Report environment details including UV project directory",
        handler="cmd_report_env_details",
    ),
    "lint": CommandConfig(
        help="Run linting and formatting",
        handler="cmd_lint",
        parser_setup=_setup_lint_parser,
    ),
    "clean": CommandConfig(
        help="Clean build artifacts and temporary files",
        handler="cmd_clean",
    ),
    "go": CommandConfig(
        help="Navigate to a Softmax Home shortcut",
        handler="cmd_go",
        pass_unknown_args=True,
    ),
    # Commands that need config but not components
    "local": CommandConfig(
        help="Local development commands",
        handler="cmd_local",
        needs_config=True,
        pass_unknown_args=True,
        add_help=False,  # Let LocalCommands handle its own help
        parser_setup=None,
    ),
    "symlink-setup": CommandConfig(
        help="Create symlink to make metta command globally available",
        handler="cmd_symlink_setup",
        needs_config=True,
        parser_setup=_setup_symlink_parser,
    ),
    # Commands that need full setup with components
    "configure": CommandConfig(
        help="Configure Metta settings",
        handler="cmd_configure",
        needs_config=True,
        needs_components=True,
        parser_setup=_setup_configure_parser,
    ),
    "install": CommandConfig(
        help="Install or update components",
        handler="cmd_install",
        needs_config=True,
        needs_components=True,
        parser_setup=_setup_install_parser,
    ),
    "status": CommandConfig(
        help="Show status of all components",
        handler="cmd_status",
        needs_config=True,
        needs_components=True,
        parser_setup=_setup_status_parser,
    ),
    "run": CommandConfig(
        help="Run component-specific commands",
        handler="cmd_run",
        needs_config=True,
        needs_components=True,
        pass_unknown_args=True,
        parser_setup=_setup_run_parser,
    ),
}


class MettaCLI:
    def __init__(self):
        self.repo_root: Path = get_repo_root()
        self._path_setup: Optional[PathSetup] = None
        self._local_commands: Optional[LocalCommands] = None
        self._book_commands: Optional[BookCommands] = None
        self._components_initialized = False

    def _init_all(self):
        """Initialize all components - used by commands that need everything."""
        if self._components_initialized:
            return

        # Import all component modules to register them with the registry

        import_all_modules_from_subpackage("metta.setup", "components")

        # Initialize core objects
        from metta.setup.local_commands import LocalCommands
        from metta.setup.symlink_setup import PathSetup

        self._path_setup = PathSetup(self.repo_root)
        self._local_commands = LocalCommands()
        self._components_initialized = True

    @property
    def path_setup(self):
        if self._path_setup is None:
            from metta.setup.symlink_setup import PathSetup

            self._path_setup = PathSetup(self.repo_root)
        return self._path_setup

    @property
    def local_commands(self):
        if self._local_commands is None:
            from metta.setup.local_commands import LocalCommands

            self._local_commands = LocalCommands()
        return self._local_commands

    @property
    def book_commands(self):
        if self._book_commands is None:
            from metta.setup.tools.book import BookCommands

            self._book_commands = BookCommands()
        return self._book_commands

    def setup_wizard(self, non_interactive: bool = False) -> None:
        from metta.setup.profiles import UserType

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

        # Add "Custom configuration" as an option
        choices = [(ut, ut.get_description()) for ut in UserType]

        # Current configuration
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

        if not self.path_setup.check_installation():
            info("You may want to run 'metta symlink-setup' to make the metta command globally available.")

    def _custom_setup(self, non_interactive: bool = False) -> None:
        from metta.setup.registry import get_all_modules

        user_type = prompt_choice(
            "Select base profile for custom configuration:",
            [(ut, ut.get_description()) for ut in UserType if ut != UserType.CUSTOM],
            default=UserType.EXTERNAL,
            non_interactive=non_interactive,
        )

        saved_settings = get_saved_settings()
        saved_settings.setup_custom_profile(user_type)

        info("\nCustomize components:")
        # Get all registered components
        all_modules = get_all_modules()

        for module in all_modules:
            current_enabled = saved_settings.is_component_enabled(module.name)

            # Use prompt_choice for yes/no
            enabled = prompt_choice(
                f"Enable {module.name} ({module.description})?",
                [(True, "Yes"), (False, "No")],
                default=current_enabled,
                current=current_enabled,
                non_interactive=non_interactive,
            )

            # Only save if different from profile default
            profile_default = (
                PROFILE_DEFINITIONS.get(user_type, {}).get("components", {}).get(module.name, {}).get("enabled", False)
            )
            if enabled != profile_default:
                saved_settings.set(f"components.{module.name}.enabled", enabled)

        success("\nCustom configuration saved.")
        info("\nRun 'metta install' to set up your environment.")

    def cmd_configure(self, args, unknown_args=None) -> None:
        if args.component:
            self.configure_component(args.component)
        elif args.profile:
            selected_user_type = UserType(args.profile)
            if selected_user_type in PROFILE_DEFINITIONS:
                saved_settings = get_saved_settings()
                saved_settings.apply_profile(selected_user_type)
                success(f"Configured as {selected_user_type.value} user.")
                info("\nRun 'metta install' to set up your environment.")
            else:
                error(f"Unknown profile: {args.profile}")
                sys.exit(1)
        else:
            self.setup_wizard(non_interactive=getattr(args, "non_interactive", False))

    def configure_component(self, component_name: str) -> None:
        from metta.setup.registry import get_all_modules
        from metta.setup.utils import error, info

        modules = get_all_modules()
        module_map = {m.name: m for m in modules}

        if not (module := module_map.get(component_name)):
            error(f"Unknown component: {component_name}")
            info(f"Available components: {', '.join(sorted(module_map.keys()))}")
            sys.exit(1)

        options = module.get_configuration_options()
        if not options:
            info(f"Component '{component_name}' has no configuration options.")
            return
        module.configure()

    def cmd_run(self, args, unknown_args=None) -> None:
        from metta.setup.registry import get_all_modules
        from metta.setup.utils import error, info

        modules = get_all_modules()
        module_map = {m.name: m for m in modules}

        if not (module := module_map.get(args.component)):
            error(f"Unknown component: {args.component}")
            info(f"Available components: {', '.join(sorted(module_map.keys()))}")
            sys.exit(1)

        # Run the component's command
        module.run(args.args)

    def cmd_install(self, args, unknown_args=None) -> None:
        from metta.setup.registry import get_all_modules, get_enabled_setup_modules
        from metta.setup.utils import error, info, success, warning

        if not get_saved_settings().exists():
            warning("No configuration found. Running setup wizard first...")
            self.setup_wizard()

        # Clean build artifacts unless --no-clean is specified
        if not args.no_clean:
            self.cmd_clean(args)

        # If specific components are requested, get all modules so we can install
        # even disabled ones (useful with --force)
        if args.components:
            modules = get_all_modules()
        else:
            modules = get_enabled_setup_modules()

        if args.components:
            only_names = args.components
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

            if module.install_once and module.check_installed() and not args.force:
                info("  -> Already installed, skipping (use --force to reinstall)\n")
                continue

            try:
                module.install(non_interactive=getattr(args, "non_interactive", False))
                print()
            except Exception as e:
                error(f"  Error: {e}\n")

        success("Installation complete!")

    def cmd_clean(self, args, unknown_args=None, verbose: bool = False) -> None:
        from metta.setup.utils import info, warning

        build_dir = self.repo_root / "build"
        if build_dir.exists():
            info("  Removing root build directory...")
            shutil.rmtree(build_dir)
        # Clean mettagrid build directories
        mettagrid_dir = self.repo_root / "mettagrid"
        for build_name in ["build-debug", "build-release"]:
            build_path = mettagrid_dir / build_name
            if build_path.exists():
                info(f"  Removing mettagrid/{build_name}...")
                shutil.rmtree(build_path)

        # Run cleanup script to remove empty directories and __pycache__
        cleanup_script = self.repo_root / "devops" / "tools" / "cleanup_repo.py"
        if cleanup_script.exists():
            cmd = [str(cleanup_script)]
            if verbose:
                cmd.append("--verbose")
            try:
                subprocess.run(cmd, cwd=str(self.repo_root), check=True)
            except subprocess.CalledProcessError as e:
                warning(f"  Cleanup script failed: {e}")

    def cmd_go(self, args, unknown_args=None) -> None:
        """Navigate to a Softmax Home shortcut URL."""
        import webbrowser

        from metta.setup.utils import error, info

        if not unknown_args:
            error("Please specify a shortcut (e.g., 'metta go g' for GitHub)")
            info("\nCommon shortcuts:")
            info("  g    - GitHub")
            info("  w    - Weights & Biases")
            info("  o    - Observatory")
            info("  d    - Datadog")
            return

        shortcut = unknown_args[0]
        url = f"https://home.softmax-research.net/{shortcut}"

        info(f"Opening {url}...")
        webbrowser.open(url)

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def cmd_symlink_setup(self, args, unknown_args=None) -> None:
        self.path_setup.setup_path(force=args.force)

    def _run_subprocess_command(self, command: str, args, unknown_args=None) -> None:
        """Run a subprocess command from the registry."""
        cmd_config = COMMAND_REGISTRY[command]
        if not cmd_config.subprocess_cmd:
            raise ValueError(f"Command {command} has no subprocess_cmd")

        cmd = list(cmd_config.subprocess_cmd)  # Copy to avoid mutation

        # For commands that pass unknown args, use sys.argv directly
        # to preserve exact argument order and avoid argparse interpretation
        if cmd_config.pass_unknown_args:
            try:
                cmd_index = sys.argv.index(command)
                remaining_args = sys.argv[cmd_index + 1 :] if cmd_index + 1 < len(sys.argv) else []
                if remaining_args:
                    cmd.extend(remaining_args)
            except ValueError:
                # Fallback to unknown_args
                if unknown_args:
                    cmd.extend(unknown_args)

        try:
            # Use check=False for commands like clip that handle their own exit codes
            check = command != "clip"
            subprocess.run(cmd, cwd=self.repo_root, check=check)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)
        except FileNotFoundError:
            print(f"Error: Command not found: {cmd[0]}", file=sys.stderr)
            if command == "clip":
                print("Run: metta install codebot", file=sys.stderr)
            sys.exit(1)

    def cmd_report_env_details(self, args, unknown_args=None) -> None:
        print(f"UV Project Directory: {self.repo_root}")
        print(f"Metta CLI Working Directory: {Path.cwd()}")

    def cmd_local(self, args, unknown_args=None) -> None:
        self.local_commands.main(unknown_args)

    def cmd_book(self, args, unknown_args=None) -> None:
        self.book_commands.main(unknown_args)

    def cmd_lint(self, args, unknown_args=None) -> None:
        files = []
        if args.staged:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                cwd=self.repo_root,
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

        # ruff check: warns
        # ruff format check: warns
        # ruff check --fix: auto-fixes
        # ruff format: auto-fixes
        if args.fix:
            check_cmd.append("--fix")
        else:
            format_cmd.append("--check")

        if files:
            for cmd in cmds:
                cmd.extend(files)

        for cmd in cmds:
            try:
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, cwd=self.repo_root, check=True)
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)

    def cmd_ci(self, args, unknown_args=None) -> None:
        """Run all Python and C++ tests for CI."""
        from metta.setup.utils import error, info, success

        # First run Python tests
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
            subprocess.run(python_test_cmd, cwd=self.repo_root, check=True)
            success("Python tests passed!")
        except subprocess.CalledProcessError as e:
            error("Python tests failed!")
            sys.exit(e.returncode)

        # Then run C++ tests
        info("\nBuilding and running C++ tests...")
        mettagrid_dir = self.repo_root / "mettagrid"

        # Check if bazel is available
        if not shutil.which("bazel"):
            error("Bazel is not installed, skipping C++ tests")
            info("To install Bazel, visit: https://bazel.build/install")
            info("On macOS: brew install bazelisk")
            info("On Ubuntu: sudo apt install bazel")
            sys.exit(1)

        # Build and run tests with Bazel
        try:
            # Run all C++ tests
            info("Running C++ tests...")
            subprocess.run(
                [
                    "bazel",
                    "test",
                    "--config=opt",
                    "--test_output=errors",
                    "//:test_stats_tracker",
                    "//:test_grid_object",
                    "//:test_mettagrid",
                    "//:test_observations",
                ],
                cwd=mettagrid_dir,
                check=True,
            )
            success("C++ tests passed!")

            # Build and run benchmarks
            info("Running C++ benchmarks...")
            subprocess.run(["bazel", "build", "--config=opt", "//benchmarks:all"], cwd=mettagrid_dir, check=True)
            subprocess.run(["bazel", "run", "//benchmarks:all"], cwd=mettagrid_dir, check=True)
            success("C++ benchmarks completed!")

        except subprocess.CalledProcessError as e:
            error(f"C++ tests/benchmarks failed: {e}")
            sys.exit(1)

        success("\nAll CI tests passed!")

    def cmd_tool(self, args, unknown_args=None) -> None:
        tool_path = self.repo_root / "tools" / f"{args.tool_name}.py"
        if not tool_path.exists():
            print(f"Error: Tool '{args.tool_name}' not found at {tool_path}", file=sys.stderr)
            sys.exit(1)

        cmd = [str(tool_path)] + (unknown_args or [])
        try:
            subprocess.run(cmd, cwd=self.repo_root, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)

    def cmd_status(self, args, unknown_args=None) -> None:
        import concurrent.futures

        from metta.setup.registry import get_all_modules
        from metta.setup.utils import error, info, spinner, success, warning

        # Get all modules first
        all_modules = get_all_modules()

        # Filter by requested components if specified
        if args.components:
            # Parse comma-separated components
            requested_components = [c.strip() for c in args.components.split(",")]
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

        # Check if any modules are applicable
        applicable_modules = [m for m in modules if m.is_enabled()]
        if not applicable_modules:
            warning("No applicable modules found.")
            return

        # Do all substantive checks upfront in parallel
        module_status = {}

        with spinner("Checking component status..."):
            # Parallelize module checks
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_module = {executor.submit(lambda m: (m.name, m.get_status()), m): m for m in modules}
                for future in concurrent.futures.as_completed(future_to_module):
                    name, status = future.result()
                    if status:
                        module_status[name] = status

        # Now do all logging/display
        max_comp_len = max(len(m.name) for m in modules) + 2
        max_conn_len = 25
        max_exp_len = 20

        # Header
        cols = [
            f"{'Component':<{max_comp_len}}",
            f"{'Installed':<10}",
            f"{'Connected As':<{max_conn_len}}",
            f"{'Expected':<{max_exp_len}}",
            "Status",
        ]
        header = " | ".join(cols)
        separator = "-" * len(header)

        info(f"\n{header}")
        info(separator)

        # Display each module using cached results
        for module in modules:
            if module.name not in module_status:
                continue

            status_data = module_status[module.name]
            installed = status_data["installed"]
            connected_as = status_data["connected_as"]
            expected = status_data["expected"]

            installed_str = "Yes" if installed else "No"
            connected_str = self._truncate(connected_as or "-", max_conn_len)
            expected_str = self._truncate(expected or "-", max_exp_len)

            # Determine status
            if not installed:
                status = "NOT INSTALLED"
                status_color = error
            elif connected_as is None:
                if expected is None:
                    # No connection needed
                    status = "OK"
                    status_color = success
                else:
                    # Should be connected but isn't
                    status = "NOT CONNECTED"
                    status_color = error
            elif expected is None:
                # Connected but no expectation
                status = "OK"
                status_color = success
            elif expected in connected_as:
                # Connected to right place
                status = "OK"
                status_color = success
            else:
                # Connected to wrong place
                status = "WRONG ACCOUNT"
                status_color = warning

            # Format row
            row_parts = [
                f"{module.name:<{max_comp_len}}",
                f"{installed_str:<10}",
                f"{connected_str:<{max_conn_len}}",
                f"{expected_str:<{max_exp_len}}",
            ]
            row = " | ".join(row_parts) + " | "
            info(row, end="")
            status_color(status)

        info(f"\n{separator}")

        all_installed = all(module_status[name]["installed"] for name in module_status)
        all_connected = all(
            (module_status[name]["connected_as"] is not None or module_status[name]["expected"] is None)
            for name in module_status
            if module_status[name]["installed"]
        )

        if all_installed:
            if all_connected:
                success("\nAll components are properly configured!")
            else:
                warning("\nSome components need authentication. Run 'metta install' to set them up.")
        else:
            warning("\nSome components are not installed. Run 'metta install' to set them up.")

        not_connected = [
            name
            for name, data in module_status.items()
            if data["installed"] and data["expected"] and data["connected_as"] is None
        ]

        # Offer to fix connection issues
        if not_connected:
            warning(f"\nComponents not connected: {', '.join(not_connected)}")
            info("This could be due to expired credentials, network issues, or broken installations.")

            if args.non_interactive:
                info(f"\nTo fix: metta install {' '.join(not_connected)} --force")
            elif sys.stdin.isatty():
                response = input("\nReinstall these components to fix connection issues? (y/n): ").strip().lower()
                if response == "y":
                    info(f"\nRunning: metta install {' '.join(not_connected)} --force")
                    subprocess.run(
                        [sys.executable, __file__, "install"] + not_connected + ["--force"], cwd=self.repo_root
                    )

        # Check for not installed components using cached results
        not_installed = [name for name, data in module_status.items() if not data["installed"]]

        if not_installed:
            warning(f"\nComponents not installed: {', '.join(not_installed)}")

            if args.non_interactive:
                info(f"\nTo fix: metta install {' '.join(not_installed)}")
            elif sys.stdin.isatty():
                response = input("\nInstall these components? (y/n): ").strip().lower()
                if response == "y":
                    info(f"\nRunning: metta install {' '.join(not_installed)}")
                    subprocess.run([sys.executable, __file__, "install"] + not_installed, cwd=self.repo_root)

    def _build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser from command registry."""
        parser = argparse.ArgumentParser(
            description="Metta Setup Tool - Configure and install development environment",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  metta configure                      # Run interactive setup wizard
  metta configure githooks             # Configure a specific component
  metta configure --profile=softmax    # Configure for Softmax employee
  metta install                        # Install all configured components
  metta install aws wandb              # Install specific components
  metta status                         # Show status of all components
  metta status --non-interactive       # Show status without prompts
  metta clean                          # Clean build artifacts
  metta symlink-setup                  # Set up symlink to make metta command globally available

  metta run githooks pre-commit        # Run component-specific commands

  metta test ...                       # Run all python unit tests
  metta pytest [args]                  # An alias "uv run pytest --benchmark-disable -n auto [args]"
  metta ci ...                         # Run all python unit tests and mettagrid c++ tests

  metta tool train run=test            # Run train.py tool with arguments
  metta tool sim policy_uri=...        # Run sim.py tool with arguments
  metta clip -e py metta               # Copy Python files to clipboard

  metta local ...                      # Commands for local development
  metta book                           # Interactive marimo notebook commands
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Create subparser for each command from registry
        for cmd_name, cmd_config in COMMAND_REGISTRY.items():
            cmd_parser = subparsers.add_parser(cmd_name, help=cmd_config.help, add_help=cmd_config.add_help)

            # Apply parser setup if provided
            if cmd_config.parser_setup:
                cmd_config.parser_setup(cmd_parser)
            elif cmd_name == "local":
                from metta.setup.local_commands import setup_local_parser

                setup_local_parser(cmd_parser)

            # Special handling for configure --profile
            if cmd_name == "configure":
                available_preset_profiles = [u.value for u in list(PROFILE_DEFINITIONS.keys())]
                cmd_parser.add_argument("--profile", choices=available_preset_profiles, help="Set user profile")

        return parser

    def main(self) -> None:
        # Build parser from registry
        parser = self._build_parser()

        # Determine which commands support unknown args
        pass_unknown_cmds = {cmd for cmd, cfg in COMMAND_REGISTRY.items() if cfg.pass_unknown_args}

        # Use parse_known_args for commands that accept unknown args
        if len(sys.argv) > 1 and sys.argv[1] in pass_unknown_cmds:
            args, unknown_args = parser.parse_known_args()
        else:
            args = parser.parse_args()
            unknown_args = []

        saved_settings = get_saved_settings()
        # Handle no command
        if not args.command:
            if not saved_settings.exists():
                print("No configuration found. Running setup wizard...\n")
                self.setup_wizard()
                return
            else:
                parser.print_help()
                return

        # Check configuration requirements based on command registry
        if args.command in COMMAND_REGISTRY:
            cmd_config = COMMAND_REGISTRY[args.command]
            if cmd_config.needs_config and args.command not in ["configure", "symlink-setup"]:
                if not saved_settings.exists():
                    print("Error: No configuration found. Please run 'metta configure' first.", file=sys.stderr)
                    sys.exit(1)
                else:
                    from metta.setup.saved_settings import CURRENT_SAVED_SETTINGS_VERSION

                    if saved_settings.config_version < CURRENT_SAVED_SETTINGS_VERSION:
                        print(
                            f"Warning: Your configuration is from an older version (v{saved_settings.config_version}).",
                            file=sys.stderr,
                        )
                        print("Please run 'metta configure' to update your configuration.", file=sys.stderr)
                        sys.exit(1)

        # Dispatch to command handler
        if args.command in COMMAND_REGISTRY:
            cmd_config = COMMAND_REGISTRY[args.command]

            # Initialize components if needed
            if cmd_config.needs_components:
                self._init_all()

            # Handle subprocess commands
            if cmd_config.subprocess_cmd:
                self._run_subprocess_command(args.command, args, unknown_args)
            # Handle method handlers
            elif cmd_config.handler:
                handler = getattr(self, cmd_config.handler)
                # Always pass both args and unknown_args for consistency
                handler(args, unknown_args)
            else:
                print(f"Error: Command {args.command} has no handler or subprocess_cmd", file=sys.stderr)
                sys.exit(1)
        else:
            parser.print_help()


def main():
    # Quick check for commands that can use fast path
    if len(sys.argv) > 1 and sys.argv[1] in COMMAND_REGISTRY:
        cmd_config = COMMAND_REGISTRY[sys.argv[1]]

        # Fast path for simple subprocess commands that don't need any setup
        if cmd_config.subprocess_cmd and not cmd_config.needs_config and not cmd_config.needs_components:
            # Direct execution without creating CLI instance
            cmd = list(cmd_config.subprocess_cmd)

            # For commands that pass unknown args, get all args after the command
            if cmd_config.pass_unknown_args:
                try:
                    cmd_index = sys.argv.index(sys.argv[1])
                    remaining_args = sys.argv[cmd_index + 1 :] if cmd_index + 1 < len(sys.argv) else []
                    if remaining_args:
                        cmd.extend(remaining_args)
                except ValueError:
                    pass

            try:
                subprocess.run(cmd, cwd=get_repo_root(), check=True)
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)
            except FileNotFoundError:
                print(f"Command not found: {cmd[0]}", file=sys.stderr)
                if sys.argv[1] == "clip":
                    print("Run: metta install codeclip", file=sys.stderr)
                sys.exit(1)
            return

    # All commands use the same initialization
    cli = MettaCLI()
    cli.main()


if __name__ == "__main__":
    main()
