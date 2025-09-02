import os
import re
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class NodejsSetup(SetupModule):
    @property
    def description(self) -> str:
        return "Node.js infrastructure - pnpm and turborepo"

    def dependencies(self) -> list[str]:
        return ["system"]  # Ensure Node.js/corepack is installed before running pnpm setup

    def _script_exists(self, script: str) -> bool:
        try:
            self.run_command(["which", script], capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def check_installed(self) -> bool:
        if not (self.repo_root / "node_modules").exists():
            return False

        if not self._check_pnpm():
            return False

        if not self._script_exists("turbo"):
            return False

        return True

    def _check_pnpm(self) -> bool:
        """Check if pnpm is working."""
        try:
            env = os.environ.copy()
            env["NODE_NO_WARNINGS"] = "1"
            result = subprocess.run(
                ["pnpm", "--version"],
                capture_output=True,
                text=True,
                env=env,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_shell_config_paths(self) -> list[tuple[str, str]]:
        """Get paths to shell configuration files.

        Returns:
            List of (shell_name, config_path) tuples
        """
        paths = []

        # zsh config - respect ZDOTDIR
        zdotdir = os.environ.get("ZDOTDIR")
        if zdotdir:
            zshrc_path = os.path.join(zdotdir, ".zshrc")
        else:
            zshrc_path = os.path.expanduser("~/.zshrc")
        paths.append(("zsh", zshrc_path))

        # bash config
        bashrc_path = os.path.expanduser("~/.bashrc")
        paths.append(("bash", bashrc_path))

        return paths

    def _check_pnpm_config_in_file(self, file_path: str, expected_pnpm_home: str) -> bool:
        """Check if a shell config file has the correct PNPM configuration.

        Args:
            file_path: Path to shell config file
            expected_pnpm_home: Expected PNPM_HOME value

        Returns:
            True if file has correct PNPM config, False otherwise
        """
        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, "r") as f:
                content = f.read()
        except (IOError, OSError):
            return False

        # Check if pnpm section exists
        if "# pnpm" not in content or "# pnpm end" not in content:
            return False

        # Check if the correct PNPM_HOME is set (handle both expanded and $HOME formats)
        expected_export_full = f'export PNPM_HOME="{expected_pnpm_home}"'
        expected_export_home = 'export PNPM_HOME="$HOME/.local/share/pnpm"'
        if expected_export_full not in content and expected_export_home not in content:
            return False

        # Check if PATH logic exists
        path_check_patterns = ['export PATH="$PNPM_HOME:$PATH"', 'case ":$PATH:" in', '*":$PNPM_HOME:"*)']
        if not any(pattern in content for pattern in path_check_patterns):
            return False

        return True

    def _print_shell_reload_instructions(self, modified_files: list[tuple[str, str]]) -> None:
        """Print instructions for reloading shell after profile modifications.

        Args:
            modified_files: List of (shell_name, config_path) tuples for modified files
        """

        if not modified_files:
            return

        success("Shell profiles have been updated with PNPM_HOME configuration!")
        info("")
        info("To apply the changes immediately, reload your shell:")

        for shell_name, _config_path in modified_files:
            if shell_name == "zsh":
                info("  For zsh: exec zsh")
            elif shell_name == "bash":
                info("  For bash: exec bash")
            else:
                info(f"  For {shell_name}: exec {shell_name}")

        info("")
        info("Or simply restart your terminal.")
        info("After reloading, global packages like 'turbo' will be available in your PATH.")

    def _update_shell_profiles_for_pnpm(self, pnpm_home: str) -> None:
        """Update shell profiles with PNPM_HOME configuration.

        Args:
            pnpm_home: Path to pnpm home directory
        """
        pnpm_config_lines = [
            "# pnpm",
            f'export PNPM_HOME="{pnpm_home}"',
            'case ":$PATH:" in',
            '  *":$PNPM_HOME:"*) ;;',
            '  *) export PATH="$PNPM_HOME:$PATH" ;;',
            "esac",
            "# pnpm end",
        ]

        modified_files = []

        for shell_name, config_path in self._get_shell_config_paths():
            # Check if file already has correct configuration
            if self._check_pnpm_config_in_file(config_path, pnpm_home):
                info(f"PNPM_HOME correctly configured in {config_path}")
                continue

            # Check if the file exists, if not, create it
            if not os.path.exists(config_path):
                # Create parent directory if needed
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w") as f:
                    f.write(f"# {shell_name} configuration\n")
                info(f"Created {config_path}")

            # Read existing content
            try:
                with open(config_path, "r") as f:
                    content = f.read()
            except (IOError, OSError):
                warning(f"Could not read {config_path}, skipping")
                continue

            # Check if there's an existing pnpm section that needs updating
            if "# pnpm" in content and "# pnpm end" in content:
                # Remove old pnpm section and add new one
                lines = content.split("\n")
                new_lines = []
                in_pnpm_section = False

                for line in lines:
                    if line.strip() == "# pnpm":
                        in_pnpm_section = True
                        continue
                    elif line.strip() == "# pnpm end":
                        in_pnpm_section = False
                        continue
                    elif not in_pnpm_section:
                        new_lines.append(line)

                # Add the new pnpm configuration
                new_content = "\n".join(new_lines)
                if not new_content.endswith("\n"):
                    new_content += "\n"
                new_content += "\n" + "\n".join(pnpm_config_lines) + "\n"

                try:
                    with open(config_path, "w") as f:
                        f.write(new_content)
                    info(f"Updated PNPM_HOME configuration in {config_path}")
                    modified_files.append((shell_name, config_path))
                except (IOError, OSError):
                    warning(f"Could not write to {config_path}")
            else:
                # No existing pnpm section, just append
                try:
                    with open(config_path, "a") as f:
                        f.write("\n" + "\n".join(pnpm_config_lines) + "\n")
                    info(f"Added PNPM_HOME configuration to {config_path}")
                    modified_files.append((shell_name, config_path))
                except (IOError, OSError):
                    warning(f"Could not write to {config_path}")

        # Provide shell reload instructions if any files were modified
        if modified_files:
            self._print_shell_reload_instructions(modified_files)

    def _setup_pnpm_environment(self, pnpm_home: str) -> None:
        """Set up PNPM_HOME and PATH in current process and shell profiles.

        Args:
            pnpm_home: Path to pnpm home directory
        """
        # Ensure directory exists
        os.makedirs(pnpm_home, exist_ok=True)

        # Set environment variables for current process
        os.environ["PNPM_HOME"] = pnpm_home
        if pnpm_home not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{pnpm_home}:{os.environ['PATH']}"

        # Update shell profiles
        self._update_shell_profiles_for_pnpm(pnpm_home)

        info(f"PNPM_HOME set to: {pnpm_home}")

    def _enable_corepack_with_cleanup(self):
        """Enable corepack, removing dead symlinks as needed."""
        for _ in range(10):
            try:
                # Try to enable corepack
                self.run_command(["corepack", "enable"], capture_output=True, check=True)
                info("Corepack enabled successfully")
                return True
            except subprocess.CalledProcessError as e:
                error_output = e.output
                # Look for EEXIST error with file path
                match = re.search(r"EEXIST: file already exists, symlink .* -> '([^']+)'", error_output)
                if match:
                    conflicting_path = match.group(1)
                    warning(f"Removing dead symlink: {conflicting_path}")
                    try:
                        if os.path.islink(conflicting_path):
                            os.remove(conflicting_path)
                            info(f"Removed conflicting file: {conflicting_path}")
                        else:
                            warning(f"Conflicting file not found: {conflicting_path}")
                    except OSError as rm_err:
                        warning(f"Failed to remove {conflicting_path}: {rm_err}")
                        raise RuntimeError(f"Cannot remove conflicting file: {conflicting_path}") from rm_err
                    continue
                else:
                    # Not an EEXIST error or couldn't parse it
                    warning(f"Corepack enable failed: {error_output}")
                    raise

        warning("Failed to enable corepack after removing dead symlinks")
        return False

    def install(self, non_interactive: bool = False) -> None:
        info("Setting up pnpm...")

        # First, determine the correct PNPM_HOME path
        pnpm_home = None

        # If PNPM_HOME already exists and is working, use it
        if os.environ.get("PNPM_HOME") and self._check_pnpm():
            pnpm_home = os.environ["PNPM_HOME"]
            info(f"Using existing PNPM_HOME: {pnpm_home}")
        else:
            # Set up PNPM_HOME BEFORE running any pnpm commands
            # Use the standard location that pnpm creates
            pnpm_home = os.path.expanduser("~/.local/share/pnpm")
            info(f"Setting up PNPM_HOME at: {pnpm_home}")

            # Set up environment and shell profiles
            self._setup_pnpm_environment(pnpm_home)

        # Now enable corepack if pnpm is not working
        if not self._check_pnpm():
            if not self._enable_corepack_with_cleanup():
                raise RuntimeError("Failed to set up pnpm via corepack")

        # Run pnpm setup to ensure shell profiles are configured correctly
        # This is safe to run multiple times and will update configs if needed
        info("Running pnpm setup to ensure shell configuration...")
        try:
            self.run_command(["pnpm", "setup"], capture_output=False)
        except subprocess.CalledProcessError as e:
            warning(f"pnpm setup failed: {e}. Continuing with manual configuration.")

        # Verify pnpm is working with correct PNPM_HOME
        if self._check_pnpm() and os.environ.get("PNPM_HOME"):
            info("Installing turbo...")
            self.run_command(["pnpm", "install", "--global", "turbo"], capture_output=False)
        else:
            warning("PNPM_HOME not properly configured, skipping global turbo install")

        info("Installing dependencies...")
        # pnpm install with frozen lockfile to avoid prompts
        self.run_command(["pnpm", "install", "--frozen-lockfile"], capture_output=False)
