#!/usr/bin/env -S uv run
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from metta.setup.config import CURRENT_CONFIG_VERSION, PROFILE_DEFINITIONS, SetupConfig, UserType
from metta.setup.registry import get_all_modules, get_applicable_modules
from metta.setup.symlink_setup import PathSetup
from metta.setup.utils import error, header, import_all_modules_from_subpackage, info, success, warning

# Import all component modules to register them with the registry
import_all_modules_from_subpackage("metta.setup", "components")


class MettaCLI:
    def __init__(self):
        self.repo_root: Path = Path(__file__).parent.parent.parent
        self.config: SetupConfig = SetupConfig()
        self.path_setup: PathSetup = PathSetup(self.repo_root)

    def setup_wizard(self) -> None:
        header("Welcome to Metta!\n\n")

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

        info("Select configuration:")
        # Dynamically generate user type options
        user_types = list(UserType)
        for i, user_type in enumerate(user_types, 1):
            info(f"{i}. {user_type.get_description()}")
        info(f"{len(user_types) + 1}. Custom configuration")

        choice = input(f"\nEnter choice (1-{len(user_types) + 1}, or press Enter to keep current): ").strip()

        if not choice and self.config.config_path.exists():
            info("Keeping current configuration.")
            return

        if choice == str(len(user_types) + 1):
            self._custom_setup()
        else:
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(user_types):
                    user_type = user_types[choice_idx]
                else:
                    user_type = UserType.EXTERNAL
            except (ValueError, IndexError):
                user_type = UserType.EXTERNAL

            self.config.apply_profile(user_type)
            success(f"\nConfigured as {user_type.value} user.")
        info("\nRun 'metta install' to set up your environment.")

        if not self.path_setup.check_installation():
            info("You may want to run 'metta symlink-setup' to make the metta command globally available.")

    def _custom_setup(self) -> None:
        info("\nSelect base profile for custom configuration:")
        # Dynamically generate user type options
        user_types = list(UserType)
        for i, user_type in enumerate(user_types, 1):
            info(f"{i}. {user_type.get_description()}")

        choice = input(f"\nEnter choice (1-{len(user_types)}): ").strip()

        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(user_types):
            user_type = user_types[choice_idx]
        else:
            raise ValueError(f"Invalid choice: {choice}")

        self.config.setup_custom_profile(user_type)

        info("\nCustomize components:")
        # Get all available components from the base profile
        base_components = PROFILE_DEFINITIONS.get(user_type, {}).get("components", {})
        for comp in base_components:
            current = self.config.is_component_enabled(comp)
            prompt = f"Enable {comp}? (y/n, current: {'y' if current else 'n'}): "
            choice = input(prompt).strip().lower()
            if choice in ["y", "n"]:
                self.config.set(f"components.{comp}.enabled", choice == "y")

        success("\nCustom configuration saved.")
        info("\nRun 'metta install' to set up your environment.")

    def cmd_configure(self, args) -> None:
        if args.profile:
            # Dynamically build profile map from UserType enum
            profile_map = {ut.value: ut for ut in UserType}
            if args.profile in profile_map:
                self.config.apply_profile(profile_map[args.profile])
                success(f"Configured as {profile_map[args.profile].value} user.")
                info("\nRun 'metta install' to set up your environment.")
            else:
                error(f"Unknown profile: {args.profile}")
                available_profiles = [ut.value for ut in UserType if ut != UserType.SOFTMAX_DOCKER]
                info(f"Available profiles: {', '.join(available_profiles)}")
                sys.exit(1)
        else:
            self.setup_wizard()

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

            if module.check_installed() and not args.force:
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
  metta configure --profile=softmax    # Configure for Softmax employee
  metta install                        # Install all configured components
  metta install aws wandb              # Install specific components
  metta status                         # Show component status
  metta clean                          # Clean build artifacts
  metta symlink-setup                  # Set up symlink to make metta command globally available
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Configure command
        configure_parser = subparsers.add_parser("configure", help="Configure Metta for your environment")
        configure_parser.add_argument(
            "--profile",
            choices=[ut.value for ut in UserType if ut != UserType.SOFTMAX_DOCKER],
            help="Set user profile (external, cloud, or softmax)",
        )

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

        args = parser.parse_args()

        # Auto-run configure if no config exists and no command given
        if not args.command and not self.config.config_path.exists():
            info("No configuration found. Running setup wizard...\n")
            self.setup_wizard()
            return

        if args.command != "configure":
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
        elif args.command == "install":
            self.cmd_install(args)
        elif args.command == "status":
            self.cmd_status(args)
        elif args.command == "clean":
            self.cmd_clean(args)
        elif args.command == "symlink-setup":
            self.cmd_symlink_setup(args)
        else:
            parser.print_help()


def main():
    cli = MettaCLI()
    cli.main()


if __name__ == "__main__":
    main()
