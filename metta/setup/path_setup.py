import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from metta.setup.utils import info, success, warning


class PathSetup:
    """Handles PATH configuration for the Metta command."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.bin_dir = repo_root / "metta" / "setup" / "installer" / "bin"
        self.metta_dir = repo_root / ".metta"
        self.metta_dir.mkdir(exist_ok=True)

        # Environment script paths in .metta directory
        self.env_script = self.metta_dir / "env"
        self.fish_env_script = self.metta_dir / "env.fish"

    def is_in_path(self) -> bool:
        """Check if metta command is accessible in PATH."""
        try:
            result = subprocess.run(["which", "metta"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                metta_path = Path(result.stdout.strip())
                # Check if it's our metta or a compatible one
                if metta_path == self.bin_dir / "metta":
                    return True
                # Check if it's a symlink to our metta
                if metta_path.is_symlink():
                    target = metta_path.resolve()
                    if target == (self.bin_dir / "metta").resolve():
                        return True
                # Check if it's the venv shim
                venv_metta = self.repo_root / ".venv" / "bin" / "metta"
                if metta_path == venv_metta and venv_metta.exists():
                    return True
        except Exception:
            pass
        return False

    def _get_home(self) -> Optional[Path]:
        """Get the user's home directory."""
        home = os.environ.get("HOME")
        if home:
            return Path(home)

        # Fallback for systems without HOME set
        try:
            import pwd

            user = os.environ.get("USER", os.getlogin())
            return Path(pwd.getpwnam(user).pw_dir)
        except Exception:
            return None

    def _create_env_scripts(self) -> None:
        """Create environment scripts for adding to PATH."""
        bin_dir_expr = str(self.bin_dir).replace(str(self._get_home()), "$HOME")

        # Create sh/bash/zsh script
        self.env_script.write_text(f"""#!/bin/sh
# add binaries to PATH if they aren't added yet
# affix colons on either side of $PATH to simplify matching
case ":${{PATH}}:" in
    *:"{bin_dir_expr}":*)
        ;;
    *)
        # Prepending path in case a system-installed binary needs to be overridden
        export PATH="{bin_dir_expr}:$PATH"
        ;;
esac
""")

        # Create fish script
        self.fish_env_script.write_text(f"""if not contains "{bin_dir_expr}" $PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "{bin_dir_expr}" $PATH
end
""")

    def _get_shell_configs(self) -> List[Tuple[Path, str]]:
        """Get list of shell configuration files to update."""
        home = self._get_home()
        if not home:
            return []

        configs = []

        # Check for various shell configs
        shell_files = [
            (".profile", "sh"),
            (".bashrc", "sh"),
            (".bash_profile", "sh"),
            (".bash_login", "sh"),
            (".zshrc", "sh"),
            (".zshenv", "sh"),
        ]

        for filename, shell_type in shell_files:
            config_path = home / filename
            if config_path.exists():
                configs.append((config_path, shell_type))

        # Fish config
        fish_config = home / ".config" / "fish" / "conf.d" / "metta.fish"
        if fish_config.parent.exists() or self._should_create_fish_config():
            configs.append((fish_config, "fish"))

        return configs

    def _should_create_fish_config(self) -> bool:
        """Check if we should create fish config directory."""
        # Check if fish is installed
        try:
            subprocess.run(["which", "fish"], capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _add_to_shell_config(self, config_path: Path, shell_type: str) -> bool:
        """Add source line to shell configuration file."""
        if shell_type == "sh":
            env_script_expr = str(self.env_script).replace(str(self._get_home()), "$HOME")
            source_line = f'. "{env_script_expr}"'
            pretty_line = f'source "{env_script_expr}"'
        else:  # fish
            env_script_expr = str(self.fish_env_script).replace(str(self._get_home()), "$HOME")
            source_line = f'source "{env_script_expr}"'
            pretty_line = source_line

        # Check if already added
        if config_path.exists():
            content = config_path.read_text()
            if source_line in content or pretty_line in content:
                return False

        # Create parent directory if needed (for fish)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Add source line
        with config_path.open("a") as f:
            # Add newline in case file doesn't end with one
            f.write("\n")
            f.write(f"{source_line}\n")

        return True

    def _add_to_github_path(self) -> bool:
        """Add to GitHub Actions PATH if running in CI."""
        github_path = os.environ.get("GITHUB_PATH")
        if github_path:
            try:
                with open(github_path, "a") as f:
                    f.write(f"{self.bin_dir}\n")
                return True
            except Exception:
                pass
        return False

    def setup_path(self, no_modify: bool = False) -> None:
        """Set up PATH configuration for metta command."""
        # Ensure wrapper script is executable
        metta_wrapper = self.bin_dir / "metta"
        metta_wrapper.chmod(0o755)

        # Check if already in PATH
        if self.is_in_path():
            success("metta is already accessible in your PATH.")
            return

        # Check if bin_dir is already in PATH
        path_dirs = os.environ.get("PATH", "").split(":")
        if str(self.bin_dir) in path_dirs:
            no_modify = True
            info("Directory is already in PATH but metta command not found.")

        if no_modify:
            warning("Skipping PATH modification.")
            info("To use metta, add this to your shell profile:")
            info(f'  export PATH="{self.bin_dir}:$PATH"')
            return

        info("Setting up PATH configuration...")

        # Create environment scripts
        self._create_env_scripts()

        # Add to CI PATH if applicable
        if self._add_to_github_path():
            info("Added to GitHub Actions PATH.")

        # Update shell configurations
        configs = self._get_shell_configs()
        if not configs:
            warning("No shell configuration files found.")
            return

        updated = False
        for config_path, shell_type in configs:
            if self._add_to_shell_config(config_path, shell_type):
                info(f"Updated {config_path}")
                updated = True

        if updated:
            info("")
            info("To add metta to your PATH, either restart your shell or run:")
            info("")
            home = self._get_home()
            if home:
                env_expr = str(self.env_script).replace(str(home), "$HOME")
                fish_env_expr = str(self.fish_env_script).replace(str(home), "$HOME")
            else:
                env_expr = str(self.env_script)
                fish_env_expr = str(self.fish_env_script)

            info(f"    source {env_expr} (sh, bash, zsh)")
            info(f"    source {fish_env_expr} (fish)")
        else:
            info("PATH already configured in shell files.")

    def check_shadowed_binaries(self) -> List[str]:
        """Check for binaries that might be shadowed by others in PATH."""
        shadowed = []
        try:
            result = subprocess.run(["which", "-a", "metta"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                paths = result.stdout.strip().split("\n")
                our_metta = str(self.bin_dir / "metta")
                for path in paths:
                    if path and path != our_metta:
                        if paths.index(path) < paths.index(our_metta) if our_metta in paths else True:
                            shadowed.append("metta")
                        break
        except Exception:
            pass

        return shadowed
