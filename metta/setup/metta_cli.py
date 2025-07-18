#!/usr/bin/env -S uv run
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from metta.setup.config import CURRENT_CONFIG_VERSION, PROFILE_DEFINITIONS, SetupConfig, UserType
from metta.setup.local_commands import LocalCommands
from metta.setup.registry import get_all_modules, get_applicable_modules
from metta.setup.symlink_setup import PathSetup
from metta.setup.utils import error, header, import_all_modules_from_subpackage, info, prompt_choice, success, warning

# Import all component modules to register them with the registry
import_all_modules_from_subpackage("metta.setup", "components")


class MettaCLI:
    def __init__(self):
        self.repo_root: Path = Path(__file__).parent.parent.parent
        self.config: SetupConfig = SetupConfig()
        self.path_setup: PathSetup = PathSetup(self.repo_root)
        self.local_commands: LocalCommands = LocalCommands(self.repo_root)

    def setup_wizard(self) -> None:
        header("Welcome to Metta!\n\n")
        info("Note: You can run 'metta configure <component>' to change component-level settings later.\n")

        if self.config.config_path.exists():
            info("Current configuration:")
            info(f"Profile: {self.config.user_type.value}")
            info(f"Mode: {'custom' if self.config.is_custom_config else 'profile'}")
            info("\nEnabled components:")
            components = self.config.get_components()
            for comp, settings in components.items():
                if settings.get("enabled"):
                    success(f"  + {comp}")
            info("\n")

        # Add "Custom configuration" as an option
        choices = [(ut, ut.get_description()) for ut in UserType]

        # Current configuration
        current_config = self.config.user_type if self.config.config_path.exists() else None

        result = prompt_choice(
            "Select configuration:",
            choices,
            current=current_config,
        )

        if result == UserType.CUSTOM:
            self._custom_setup()
        else:
            self.config.apply_profile(result)
            success(f"\nConfigured as {result.value} user.")
        info("\nRun 'metta install' to set up your environment.")

        if not self.path_setup.check_installation():
            info("You may want to run 'metta symlink-setup' to make the metta command globally available.")

    def _custom_setup(self) -> None:
        user_type = prompt_choice(
            "Select base profile for custom configuration:",
            [(ut, ut.get_description()) for ut in UserType if ut != UserType.CUSTOM],
            default=UserType.EXTERNAL,
        )

        self.config.setup_custom_profile(user_type)

        info("\nCustomize components:")
        # Get all registered components
        all_modules = get_all_modules(self.config)
        # Sort by name for consistent ordering
        all_modules.sort(key=lambda m: m.name)

        for module in all_modules:
            current_enabled = self.config.is_component_enabled(module.name)

            # Use prompt_choice for yes/no
            enabled = prompt_choice(
                f"Enable {module.name} ({module.description})?",
                [(True, "Yes"), (False, "No")],
                default=current_enabled,
                current=current_enabled,
            )

            # Only save if different from profile default
            profile_default = (
                PROFILE_DEFINITIONS.get(user_type, {}).get("components", {}).get(module.name, {}).get("enabled", False)
            )
            if enabled != profile_default:
                self.config.set(f"components.{module.name}.enabled", enabled)

        success("\nCustom configuration saved.")
        info("\nRun 'metta install' to set up your environment.")

    def cmd_configure(self, args) -> None:
        if args.component:
            self.configure_component(args.component)
        elif args.profile:
            selected_user_type = UserType(args.profile)
            if selected_user_type in PROFILE_DEFINITIONS:
                self.config.apply_profile(selected_user_type)
                success(f"Configured as {selected_user_type.value} user.")
                info("\nRun 'metta install' to set up your environment.")
            else:
                error(f"Unknown profile: {args.profile}")
                sys.exit(1)
        else:
            self.setup_wizard()

    def configure_component(self, component_name: str) -> None:
        modules = get_all_modules(self.config)
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

    def cmd_run(self, args) -> None:
        """Run component-specific commands."""
        modules = get_all_modules(self.config)
        module_map = {m.name: m for m in modules}

        if not (module := module_map.get(args.component)):
            error(f"Unknown component: {args.component}")
            info(f"Available components: {', '.join(sorted(module_map.keys()))}")
            sys.exit(1)

        # Run the component's command
        module.run(args.args)

    def cmd_install(self, args) -> None:
        if not self.config.config_path.exists():
            warning("No configuration found. Running setup wizard first...")
            self.setup_wizard()

        # Clean build artifacts unless --no-clean is specified
        if not args.no_clean:
            self.cmd_clean(args)

        # If specific components are requested, get all modules so we can install
        # even disabled ones (useful with --force)
        if args.components:
            modules = get_all_modules(self.config)
        else:
            modules = get_applicable_modules(self.config)

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
                module.install()
                print()
            except Exception as e:
                error(f"  Error: {e}\n")

        success("Installation complete!")

    def cmd_clean(self, args) -> None:
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
            try:
                subprocess.run([sys.executable, str(cleanup_script)], check=True)
            except subprocess.CalledProcessError as e:
                warning(f"  Cleanup script failed: {e}")

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def cmd_symlink_setup(self, args) -> None:
        self.path_setup.setup_path(force=args.force)

    def cmd_pytest(self, args) -> None:
        cmd = ["pytest"] + args
        try:
            subprocess.run(cmd, cwd=self.repo_root, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)

    def cmd_shell(self) -> None:
        subprocess.run(["uv", "run", "metta/setup/shell.py"], cwd=self.repo_root, check=True)

    def cmd_local(self, args) -> None:
        """Handle local development commands."""
        if hasattr(args, "local_command") and args.local_command:
            if args.local_command == "build-docker-img":
                self.local_commands.build_docker_img(args)
            else:
                error(f"Unknown local command: {args.local_command}")
                sys.exit(1)
        else:
            # Show help for local subcommand
            args.local_parser.print_help()

    def cmd_lint(self, args) -> None:
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

        check_cmd = ["uv", "run", "ruff", "check"]
        format_cmd = ["uv", "run", "ruff", "format"]
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
                info(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, cwd=self.repo_root, check=True)
            except subprocess.CalledProcessError as e:
                sys.exit(e.returncode)

    def cmd_tool(self, tool_name: str, args: list[str]) -> None:
        tool_path = self.repo_root / "tools" / f"{tool_name}.py"
        if not tool_path.exists():
            error(f"Tool '{tool_name}' not found at {tool_path}")
            sys.exit(1)

        cmd = [str(tool_path)] + args
        try:
            # Prefixing with `uv run` should not be necessary
            # because PATH is inherited and tools have uv shebangs
            subprocess.run(cmd, cwd=self.repo_root, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)

    def cmd_docs(self, args) -> None:
        """Handle documentation commands with MkDocs."""
        if args.docs_command == "serve":
            info("Starting documentation server at http://localhost:8000")
            info("Press Ctrl+C to stop")
            cmd = ["uv", "run", "mkdocs", "serve"]
            if args.port:
                cmd.extend(["--dev-addr", f"localhost:{args.port}"])
        elif args.docs_command == "build":
            info("Building documentation site...")
            cmd = ["uv", "run", "mkdocs", "build"]
            if args.clean:
                cmd.append("--clean")
        elif args.docs_command == "deploy":
            info("Deploying documentation to GitHub Pages...")
            cmd = ["uv", "run", "mkdocs", "gh-deploy"]
            if args.force:
                cmd.append("--force")
        else:
            args.docs_parser.print_help()
            return

        try:
            subprocess.run(cmd, cwd=self.repo_root, check=True)
            if args.docs_command == "build":
                success("Documentation built in ./site/")
        except subprocess.CalledProcessError as e:
            error(f"Documentation command failed: {e}")
            sys.exit(e.returncode)

    def cmd_status(self, _args) -> None:
        """Show status of all components."""
        modules = get_all_modules(self.config)

        if not modules:
            warning("No modules found.")
            return

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

        # Check each module
        for module in modules:
            if not module.is_applicable():
                continue

            # Check installed
            installed = module.check_installed()
            installed_str = "Yes" if installed else "No"

            # Check connection
            connected_as = module.check_connected_as() if installed else None
            connected_str = self._truncate(connected_as or "-", max_conn_len)

            # Get expected connection
            expected = self.config.get_expected_connection(module.name)
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

        # Summary
        if all(m.check_installed() for m in modules if m.is_applicable()):
            if all(
                (m.check_connected_as() is not None or self.config.get_expected_connection(m.name) is None)
                for m in modules
                if m.is_applicable() and m.check_installed()
            ):
                success("\nAll components are properly configured!")
            else:
                warning("\nSome components need authentication. Run 'metta install' to set them up.")
        else:
            warning("\nSome components are not installed. Run 'metta install' to set them up.")

        # Collect components with connection issues
        not_connected = []
        for module in modules:
            if module.is_applicable() and module.check_installed():
                expected = self.config.get_expected_connection(module.name)
                if expected and module.check_connected_as() is None:
                    not_connected.append(module.name)

        # Offer to fix connection issues
        if not_connected and sys.stdin.isatty():
            warning(f"\nComponents not connected: {', '.join(not_connected)}")
            info("This could be due to expired credentials, network issues, or broken installations.")
            response = input("\nReinstall these components to fix connection issues? (y/n): ").strip().lower()
            if response == "y":
                info(f"\nRunning: ./metta.sh install {' '.join(not_connected)} --force")
                subprocess.run([sys.executable, __file__, "install"] + not_connected + ["--force"], cwd=self.repo_root)

        # Check for not installed components
        not_installed = []
        for module in modules:
            if module.is_applicable() and not module.check_installed():
                not_installed.append(module.name)

        if not_installed and sys.stdin.isatty():
            warning(f"\nComponents not installed: {', '.join(not_installed)}")
            response = input("\nInstall these components? (y/n): ").strip().lower()
            if response == "y":
                info(f"\nRunning: ./metta.sh install {' '.join(not_installed)}")
                subprocess.run([sys.executable, __file__, "install"] + not_installed, cwd=self.repo_root)

    def main(self) -> None:
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
  metta status                         # Show component status
  metta clean                          # Clean build artifacts
  metta symlink-setup                  # Set up symlink to make metta command globally available

  metta run githooks pre-commit        # Run component-specific commands

  metta test ...                       # Run python unit tests
  metta test-changed ...               # Run python unit tests affected by changes

  metta tool train run=test            # Run train.py tool with arguments
  metta tool sim policy_uri=...        # Run sim.py tool with arguments

  metta docs serve                     # Serve documentation locally with live reload
  metta docs build                     # Build static documentation site
  metta docs deploy                    # Deploy documentation to GitHub Pages
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Configure command
        configure_parser = subparsers.add_parser("configure", help="Configure Metta for your environment")
        configure_parser.add_argument(
            "component",
            nargs="?",
            help="Specific component to configure (e.g., githooks). If omitted, runs the setup wizard.",
        )
        available_preset_profiles = [u.value for u in list(PROFILE_DEFINITIONS.keys())]
        configure_parser.add_argument(
            "--profile",
            choices=available_preset_profiles,
            help=f"Set user profile (available: {', '.join(available_preset_profiles)})",
        )

        # Run command
        run_parser = subparsers.add_parser("run", help="Run component-specific commands")
        run_parser.add_argument("component", help="Component to run command for (e.g., githooks)")
        run_parser.add_argument("args", nargs="*", help="Arguments to pass to the component")

        # Install command
        install_parser = subparsers.add_parser("install", help="Install configured components")
        install_parser.add_argument(
            "components",
            nargs="*",
            help=(
                "Specific components to install (e.g., mettascope aws). "
                "If omitted, installs all configured components. "
                "Note: 'system' and 'core' are always included as they are essential."
            ),
        )
        install_parser.add_argument("--force", action="store_true", help="Force reinstall even if already installed")
        install_parser.add_argument(
            "--no-clean", action="store_true", help="Skip cleaning build artifacts before installation"
        )

        # Status command
        subparsers.add_parser("status", help="Show installation and authentication status of all components")

        # Clean command
        subparsers.add_parser("clean", help="Clean build artifacts and temporary files")
        # Symlink setup command
        symlink_parser = subparsers.add_parser(
            "symlink-setup", help="Create symlink to make metta command globally available"
        )
        symlink_parser.add_argument("--force", action="store_true", help="Replace existing metta command if it exists")

        # Test commands
        subparsers.add_parser("test", help="Run python unit tests")
        subparsers.add_parser("test-changed", help="Run python unit tests affected by changes")

        # Lint command
        lint_parser = subparsers.add_parser("lint", help="Run linting and formatting")
        lint_parser.add_argument(
            "--fix",
            action="store_true",
            help="Apply fixes automatically. If not specified, just checks for issues.",
        )
        lint_parser.add_argument("--staged", action="store_true", help="Only lint staged files")

        # Tool command
        tool_parser = subparsers.add_parser("tool", help="Run a tool from the tools/ directory")
        tool_parser.add_argument("tool_name", help="Name of the tool to run (e.g., 'train', 'sim', 'analyze')")

        # Shell command
        subparsers.add_parser("shell", help="Start an IPython shell with Metta imports")

        # Docs command
        docs_parser = subparsers.add_parser("docs", help="Manage documentation with MkDocs")
        docs_subparsers = docs_parser.add_subparsers(dest="docs_command", help="Documentation commands")

        # Docs subcommands
        serve_parser = docs_subparsers.add_parser("serve", help="Serve documentation locally with live reload")
        serve_parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")

        build_parser = docs_subparsers.add_parser("build", help="Build static documentation site")
        build_parser.add_argument("--clean", action="store_true", help="Clean build directory first")

        deploy_parser = docs_subparsers.add_parser("deploy", help="Deploy documentation to GitHub Pages")
        deploy_parser.add_argument("--force", action="store_true", help="Force deployment")

        # Store docs_parser for help display
        docs_parser.set_defaults(docs_parser=docs_parser)

        # Local command
        local_parser = subparsers.add_parser("local", help="Local development commands")
        local_subparsers = local_parser.add_subparsers(dest="local_command", help="Available local commands")

        # Local subcommands
        local_subparsers.add_parser("build-docker-img", help="Build local development Docker image")

        # Store local_parser for help display
        local_parser.set_defaults(local_parser=local_parser)

        # Use parse_known_args to handle unknown arguments for test commands
        args, unknown_args = parser.parse_known_args()

        if args.command not in ["test", "test-changed", "tool"]:
            if unknown_args:
                parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")

        # Auto-run configure if no config exists and no command given
        if not args.command and not self.config.config_path.exists():
            info("No configuration found. Running setup wizard...\n")
            self.setup_wizard()
            return

        # Check if configuration is required for this command
        # Allow configure, symlink-setup, and local to run without config
        if args.command not in ["configure", "symlink-setup", "local"]:
            if not self.config.config_path.exists():
                error("No configuration found. Please run 'metta configure' first.")
                sys.exit(1)
            elif self.config.config_version < CURRENT_CONFIG_VERSION:
                # Old config format detected
                warning(f"Your configuration is from an older version (v{self.config.config_version}).")
                info("Please run 'metta configure' to update your configuration.")
                sys.exit(1)

        # Dispatch to command handler
        if args.command == "configure":
            self.cmd_configure(args)
        elif args.command == "run":
            self.cmd_run(args)
        elif args.command == "install":
            self.cmd_install(args)
        elif args.command == "status":
            self.cmd_status(args)
        elif args.command == "clean":
            self.cmd_clean(args)
        elif args.command == "symlink-setup":
            self.cmd_symlink_setup(args)
        elif args.command == "test":
            self.cmd_pytest(unknown_args)
        elif args.command == "test-changed":
            self.cmd_pytest(unknown_args + ["--testmon"])
        elif args.command == "tool":
            self.cmd_tool(args.tool_name, unknown_args)
        elif args.command == "lint":
            self.cmd_lint(args)
        elif args.command == "shell":
            self.cmd_shell()
        elif args.command == "docs":
            self.cmd_docs(args)
        elif args.command == "local":
            self.cmd_local(args)
        else:
            parser.print_help()


def main():
    cli = MettaCLI()
    cli.main()


if __name__ == "__main__":
    main()
