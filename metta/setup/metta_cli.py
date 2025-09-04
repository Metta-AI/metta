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

import gitta
from metta.common.util.fs import get_repo_root
from metta.setup.local_commands import app as local_app
from metta.setup.profiles import PROFILE_DEFINITIONS, UserType
from metta.setup.saved_settings import get_saved_settings
from metta.setup.symlink_setup import app as symlink_app
from metta.setup.tools.book import app as book_app
from metta.setup.utils import error, header, import_all_modules_from_subpackage, info, prompt_choice, success

console = Console()
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


def collect_missing_dependencies(module_map, all_modules_dict):
    """Recursively collect all missing dependencies for installation.

    This function performs a depth-first traversal of the dependency graph to find
    all transitive dependencies that are required but not currently in the install set.
    This ensures that when components are installed, all their dependencies are also
    installed, preventing runtime failures.

    Example:
        >>> # User requests only 'notebookwidgets'
        >>> requested = {'notebookwidgets': notebook_module}
        >>> all_available = {
        ...     'notebookwidgets': notebook_module,  # depends on nodejs
        ...     'nodejs': nodejs_module,             # depends on system
        ...     'system': system_module              # no dependencies
        ... }
        >>> missing = collect_missing_dependencies(requested, all_available)
        >>> # Returns {'nodejs', 'system'} - both transitive dependencies

    Dependency Resolution Strategy:
        - Breadth-first collection: Visits each node once to prevent infinite loops
        - Transitive closure: Follows dependency chains to their roots
        - Auto-inclusion: Automatically adds missing deps to prevent install failures

    Args:
        module_map: Dict mapping module names to modules selected for installation
        all_modules_dict: Dict mapping all available module names to modules

    Returns:
        Set of missing dependency names that must be added to the install set
    """
    missing_deps = set()

    def collect_dependencies(module_names, visited=None):
        if visited is None:
            visited = set()

        for name in module_names:
            if name in visited:
                continue
            visited.add(name)

            if name in module_map:
                module = module_map[name]
            elif name in all_modules_dict:
                # Handle missing dependencies that we need to collect
                module = all_modules_dict[name]
            else:
                continue

            for dep_name in module.dependencies():
                if dep_name not in module_map and dep_name in all_modules_dict:
                    missing_deps.add(dep_name)
                collect_dependencies([dep_name], visited)

    collect_dependencies(list(module_map.keys()))
    return missing_deps


def topological_sort_parallel(modules_dict, deps):
    """Organize modules into parallel installation batches using topological sorting.

    This function implements Kahn's algorithm for topological sorting with a parallel
    optimization twist. Instead of producing a single linear order, it groups modules
    into batches where all modules within a batch can be installed simultaneously
    without violating dependency constraints.

    Installation Priority Strategy:
        1. Dependencies-first: Modules with no dependencies install first
        2. Parallel opportunities: Independent modules install concurrently
        3. Respect constraints: Dependent modules wait for their dependencies
        4. Minimize batches: Maximize parallelism within dependency constraints

    Example:
        >>> # Dependency graph: A->C, B->C (A and B both depend on C)
        >>> modules = {'A': mod_a, 'B': mod_b, 'C': mod_c, 'D': mod_d}
        >>> deps = {'A': ['C'], 'B': ['C'], 'C': [], 'D': []}
        >>> batches = topological_sort_parallel(modules, deps)
        >>> # Returns [['C', 'D'], ['A', 'B']]
        >>> # Batch 1: C and D install in parallel (no dependencies)
        >>> # Batch 2: A and B install in parallel (both depend on C from batch 1)

    Performance Benefits:
        - ThreadPoolExecutor can install all modules in each batch concurrently
        - Reduces total installation time from O(n) to O(dependency_depth)
        - 4-core systems can install up to 4 components simultaneously per batch

    Circular Dependency Handling:
        - Incomplete batches indicate circular dependencies (deadlock)
        - All remaining modules will have in_degree > 0 when queue is empty

    Args:
        modules_dict: Dict mapping module names to module instances
        deps: Dict mapping module names to their dependency name lists

    Returns:
        List of batches, where each batch is a list of module names that can
        be installed in parallel. Dependencies are satisfied by prior batches.
    """
    from collections import defaultdict, deque

    in_degree = defaultdict(int)
    for module in modules_dict:
        for _dep in deps.get(module, []):
            in_degree[module] += 1

    ready_queue = deque([m for m in modules_dict if in_degree[m] == 0])
    sorted_batches = []

    while ready_queue:
        current_batch = list(ready_queue)
        sorted_batches.append(current_batch)
        ready_queue.clear()

        for module in current_batch:
            for other_module, other_deps in deps.items():
                if module in other_deps:
                    in_degree[other_module] -= 1
                    if in_degree[other_module] == 0:
                        ready_queue.append(other_module)

    return sorted_batches


class MettaCLI:
    def __init__(self):
        self.repo_root: Path = get_repo_root()
        self._components_initialized = False

    def _init_all(self):
        """Initialize all components - used by commands that need everything."""
        if self._components_initialized:
            return

        import_all_modules_from_subpackage("metta.setup", "components")
        self._components_initialized = True

    def setup_wizard(self, non_interactive: bool = False):
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
        import concurrent.futures
        import threading
        from collections import defaultdict, deque

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

        if not modules:
            info("No modules to install.")
            return

        # Build dependency graph and ensure all dependencies are included
        module_map = {m.name: m for m in modules}
        all_modules_dict = {m.name: m for m in get_all_modules()}

        # Add missing dependencies to install set
        missing_deps = collect_missing_dependencies(module_map, all_modules_dict)

        # Add missing dependencies to modules list
        if missing_deps:
            info(f"Adding missing dependencies: {', '.join(sorted(missing_deps))}")
            for dep_name in missing_deps:
                if dep_name in all_modules_dict:
                    modules.append(all_modules_dict[dep_name])
                    module_map[dep_name] = all_modules_dict[dep_name]

        # Build final dependency graph
        dependencies = {}
        for module in modules:
            deps = module.dependencies()
            # All dependencies should now be in our install set
            dependencies[module.name] = deps

        # Perform topological sort with parallelization opportunities
        install_batches = topological_sort_parallel(module_map, dependencies)

        if not install_batches:
            info("No modules to install.")
            return

        total_modules = sum(len(batch) for batch in install_batches)
        info("\nDependency analysis:")
        for module_name, deps in dependencies.items():
            if deps:
                info(f"  {module_name} depends on: {', '.join(deps)}")

        info(f"\nInstalling {total_modules} components in {len(install_batches)} parallel batches:")
        for i, batch in enumerate(install_batches):
            info(f"  Batch {i + 1}: {', '.join(batch)}")
        info("")

        # Thread-safe progress tracking
        install_lock = threading.Lock()
        installed_count = 0

        def install_module(module):
            nonlocal installed_count

            # Quick logging without holding lock during expensive operations
            with install_lock:
                info(f"[{module.name}] {module.description}")

            # Do expensive check_installed() in parallel (outside lock)
            if module.install_once and not args.force:
                try:
                    if module.check_installed():
                        with install_lock:
                            info("  -> Already installed, skipping (use --force to reinstall)")
                            installed_count += 1
                        return True
                except Exception:
                    # If check fails, proceed with installation
                    pass

            try:
                # Do the actual installation in parallel (most expensive part)
                module.install(non_interactive=getattr(args, "non_interactive", False))
                with install_lock:
                    info(f"  -> {module.name} installation complete")
                    installed_count += 1
                return True
            except Exception as e:
                with install_lock:
                    error(f"  -> {module.name} failed: {e}")
                return False

        # Install in batches, parallelizing within each batch
        failed_modules = []
        for batch_idx, batch in enumerate(install_batches):
            if len(batch) == 1:
                # Single module - install directly to avoid threading overhead
                info(f"Batch {batch_idx + 1}/{len(install_batches)}: Installing {batch[0]} (single module)")
                module = module_map[batch[0]]
                if not install_module(module):
                    failed_modules.append(module.name)
            else:
                # Multiple modules - install in parallel
                info(f"Batch {batch_idx + 1}/{len(install_batches)}: Installing {len(batch)} components in parallel...")

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), 4)) as executor:
                    batch_modules = [module_map[name] for name in batch]
                    future_to_module = {executor.submit(install_module, module): module for module in batch_modules}

                    for future in concurrent.futures.as_completed(future_to_module):
                        module = future_to_module[future]
                        try:
                            success_result = future.result()
                            if not success_result:
                                failed_modules.append(module.name)
                        except Exception as e:
                            failed_modules.append(module.name)
                            with install_lock:
                                error(f"  -> {module.name} failed with exception: {e}")

            print()  # Add spacing between batches

        if failed_modules:
            error(f"Installation completed with {len(failed_modules)} failures: {', '.join(failed_modules)}")
        else:
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
    """Install or update components with parallel execution and dependency resolution."""
    import concurrent.futures
    import threading
    from collections import defaultdict, deque

    from metta.setup.registry import get_all_modules, get_enabled_setup_modules
    from metta.setup.utils import error, info, success, warning

    cli._init_all()

    if not get_saved_settings().exists():
        warning("No configuration found. Running setup wizard first...")
        cli.setup_wizard()

    # Clean build artifacts unless --no-clean is specified
    if not no_clean:
        cmd_clean()

    # If specific components are requested, get all modules so we can install
    # even disabled ones (useful with --force)
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

    if not modules:
        info("No modules to install.")
        return

    # Build dependency graph and ensure all dependencies are included
    module_map = {m.name: m for m in modules}
    all_modules_dict = {m.name: m for m in get_all_modules()}

    # Add missing dependencies to install set
    missing_deps = collect_missing_dependencies(module_map, all_modules_dict)

    # Add missing dependencies to modules list
    if missing_deps:
        info(f"Adding missing dependencies: {', '.join(sorted(missing_deps))}")
        for dep_name in missing_deps:
            if dep_name in all_modules_dict:
                modules.append(all_modules_dict[dep_name])
                module_map[dep_name] = all_modules_dict[dep_name]

    # Build final dependency graph
    dependencies = {}
    for module in modules:
        deps = module.dependencies()
        # All dependencies should now be in our install set
        dependencies[module.name] = deps

    # Perform topological sort with parallelization opportunities
    def topological_sort_parallel(modules_dict, deps):
        """Sort modules respecting dependencies, grouping independent modules together."""
        in_degree = defaultdict(int)
        for module in modules_dict:
            for _dep in deps.get(module, []):
                in_degree[module] += 1

        # Start with modules that have no dependencies
        ready_queue = deque([m for m in modules_dict if in_degree[m] == 0])
        sorted_batches = []

        while ready_queue:
            # All modules in ready_queue can be installed in parallel
            current_batch = list(ready_queue)
            sorted_batches.append(current_batch)
            ready_queue.clear()

            # Remove current batch from graph and update in_degree
            for module in current_batch:
                for other_module, other_deps in deps.items():
                    if module in other_deps:
                        in_degree[other_module] -= 1
                        if in_degree[other_module] == 0:
                            ready_queue.append(other_module)

        return sorted_batches

    install_batches = topological_sort_parallel(module_map, dependencies)

    if not install_batches:
        info("No modules to install.")
        return

    total_modules = sum(len(batch) for batch in install_batches)
    info("\nDependency analysis:")
    for module_name, deps in dependencies.items():
        if deps:
            info(f"  {module_name} depends on: {', '.join(deps)}")

    info(f"\nInstalling {total_modules} components in {len(install_batches)} parallel batches:")
    for i, batch in enumerate(install_batches):
        info(f"  Batch {i + 1}: {', '.join(batch)}")
    info("")

    # Thread-safe progress tracking
    install_lock = threading.Lock()
    installed_count = 0

    def install_module(module):
        nonlocal installed_count

        # Quick logging without holding lock during expensive operations
        with install_lock:
            info(f"[{module.name}] {module.description}")

        # Do expensive check_installed() in parallel (outside lock)
        if module.install_once and not force:
            try:
                if module.check_installed():
                    with install_lock:
                        info("  -> Already installed, skipping (use --force to reinstall)")
                        installed_count += 1
                    return True
            except Exception:
                # If check fails, proceed with installation
                pass

        try:
            # Do the actual installation in parallel (most expensive part)
            module.install(non_interactive=non_interactive)
            with install_lock:
                info(f"  -> {module.name} installation complete")
                installed_count += 1
            return True
        except Exception as e:
            with install_lock:
                error(f"  -> {module.name} failed: {e}")
            return False

    # Install in batches, parallelizing within each batch
    failed_modules = []
    for batch_idx, batch in enumerate(install_batches):
        if len(batch) == 1:
            # Single module - install directly to avoid threading overhead
            info(f"Batch {batch_idx + 1}/{len(install_batches)}: Installing {batch[0]} (single module)")
            module = module_map[batch[0]]
            if not install_module(module):
                failed_modules.append(module.name)
        else:
            # Multiple modules - install in parallel
            info(f"Batch {batch_idx + 1}/{len(install_batches)}: Installing {len(batch)} components in parallel...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), 4)) as executor:
                batch_modules = [module_map[name] for name in batch]
                future_to_module = {executor.submit(install_module, module): module for module in batch_modules}

                for future in concurrent.futures.as_completed(future_to_module):
                    module = future_to_module[future]
                    try:
                        success_result = future.result()
                        if not success_result:
                            failed_modules.append(module.name)
                    except Exception as e:
                        failed_modules.append(module.name)
                        with install_lock:
                            error(f"  -> {module.name} failed with exception: {e}")

        print()  # Add spacing between batches

    if failed_modules:
        error(f"Installation completed with {len(failed_modules)} failures: {', '.join(failed_modules)}")
    else:
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
    from metta.setup.utils import info, warning

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

    console.print(table)

    all_installed = all(module_status[name]["installed"] for name in module_status)
    all_connected = all(
        (module_status[name]["connected_as"] is not None or module_status[name]["expected"] is None)
        for name in module_status
        if module_status[name]["installed"]
    )

    if all_installed:
        if all_connected:
            console.print("[green]All components are properly configured![/green]")
        else:
            console.print("[yellow]Some components need authentication. Run 'metta install' to set them up.[/yellow]")
    else:
        console.print("[yellow]Some components are not installed. Run 'metta install' to set them up.[/yellow]")

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
    from metta.setup.utils import error, info

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
    from metta.setup.utils import info, warning

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
            console.print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=cli.repo_root, check=True)
        except subprocess.CalledProcessError as e:
            raise typer.Exit(e.returncode) from e


# CI command
@app.command(name="ci", help="Run all Python unit tests and all Mettagrid C++ tests")
def cmd_ci():
    """Run all Python unit tests and all Mettagrid C++ tests."""
    from metta.setup.utils import error, info, success

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
        console.print(f"[red]Error: Tool '{tool_name}' not found at {tool_path}[/red]")
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

    from metta.setup.utils import error, info

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
    console.print(f"UV Project Directory: {cli.repo_root}")
    console.print(f"Metta CLI Working Directory: {Path.cwd()}")
    if branch := gitta.get_current_branch():
        console.print(f"Git Branch: {branch}")
    if commit := gitta.get_current_commit():
        console.print(f"Git Commit: {commit}")


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
        console.print("[red]Error: Command not found: codeclip[/red]")
        console.print("Run: metta install codebot")
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
