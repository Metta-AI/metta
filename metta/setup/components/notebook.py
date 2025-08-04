import json
import subprocess
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, success, warning


@register_module
class NotebookSetup(SetupModule):
    """
    Jupyter notebook support component.

    This component handles installation and registration of the Metta Jupyter kernel.
    It uses Jupyter's built-in discovery mechanism to find kernels, making it
    cross-platform compatible (works on macOS, Linux, Windows).

    The kernel is registered as 'metta-venv-kernel' with display name 'metta-venv-kernel',
    allowing users to select it in Jupyter notebooks and labs.
    """

    # Kernel configuration constants
    KERNEL_NAME = "metta-venv-kernel"
    DISPLAY_NAME = "metta-venv-kernel"

    @property
    def description(self) -> str:
        return "Jupyter notebook support with kernel registration"

    def is_applicable(self) -> bool:
        # This component is always applicable - all users need notebook support
        return True

    def _get_kernel_path(self, kernel_name: str) -> Path | None:
        """
        Get kernel path using Jupyter's discovery mechanism.

        This method uses 'jupyter kernelspec list --json' to find kernels,
        which is cross-platform and handles different OS kernel locations automatically.

        Args:
            kernel_name: Name of the kernel to find (e.g., 'metta-notebook-kernel')

        Returns:
            Path to kernel directory if found, None otherwise
        """
        try:
            # Use Jupyter's built-in discovery to find kernels
            # This works on all platforms (macOS, Linux, Windows)
            result = self.run_command(["jupyter", "kernelspec", "list", "--json"], check=False)
            if result.returncode == 0:
                kernels = json.loads(result.stdout)
                if kernel_name in kernels["kernelspecs"]:
                    return Path(kernels["kernelspecs"][kernel_name]["resource_dir"])
        except Exception:
            # If Jupyter discovery fails, return None
            # This could happen if jupyter command is not available
            pass
        return None

    def _validate_kernel_spec(self, kernel_path: Path) -> bool:
        """
        Validate that a kernel spec points to our uv environment.

        This checks that the kernel's argv points to either:
        1. A uv command (e.g., 'uv run python')
        2. Our .venv directory (e.g., '/path/to/.venv/bin/python')

        Args:
            kernel_path: Path to kernel directory containing kernel.json

        Returns:
            True if kernel is valid, False otherwise
        """
        kernel_json = kernel_path / "kernel.json"
        if not kernel_json.exists():
            return False

        try:
            with open(kernel_json) as f:
                kernel_spec = json.load(f)

            # Check that kernel spec has required argv field
            if "argv" not in kernel_spec:
                return False

            argv = kernel_spec["argv"]
            if len(argv) < 2:
                return False

            # Validate that kernel points to our environment
            # It should either use 'uv' or point to our '.venv' directory
            uses_uv = any("uv" in arg for arg in argv)
            uses_venv = any(".venv" in arg for arg in argv)

            return uses_uv or uses_venv

        except Exception:
            return False

    def check_installed(self) -> bool:
        """
        Check if ipykernel is installed and kernel is registered.

        This method performs two checks:
        1. Verifies ipykernel package is available in our environment
        2. Verifies our kernel is registered and points to our environment

        Returns:
            True if both checks pass, False otherwise
        """
        try:
            # Step 1: Check if ipykernel package is available
            result = self.run_command(
                ["uv", "run", "python", "-c", "import ipykernel; print('ipykernel available')"],
                check=False,
            )
            if result.returncode != 0:
                return False

            # Step 2: Check if our kernel is registered using Jupyter discovery
            kernel_path = self._get_kernel_path(self.KERNEL_NAME)
            if kernel_path is None:
                return False

            # Step 3: Validate that the kernel spec is correct
            return self._validate_kernel_spec(kernel_path)

        except Exception as e:
            warning(f"Error checking notebook installation: {e}")
            return False

    def install(self) -> None:
        """
        Install ipykernel and register the kernel.

        This method performs the following steps:
        1. Syncs dependencies to ensure ipykernel is available
        2. Removes any existing kernel with the same name (idempotent)
        3. Registers a new kernel using ipykernel install

        The kernel is registered as 'metta-venv-kernel' with display name 'metta-venv-kernel',
        making it available in Jupyter notebooks and labs.
        """
        try:
            # Step 1: Ensure dependencies are synced
            self.run_command(["uv", "sync"])
            success("✓ Dependencies synced")

            # Step 2: Remove any existing kernel with this name (idempotent)
            # Use Jupyter discovery to find existing kernel
            existing_kernel_path = self._get_kernel_path(self.KERNEL_NAME)
            if existing_kernel_path and existing_kernel_path.exists():
                import shutil

                shutil.rmtree(existing_kernel_path)
                success("✓ Removed existing kernel")

            # Step 3: Register the kernel with exact name
            kernel_name = self.KERNEL_NAME
            display_name = self.DISPLAY_NAME

            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "ipykernel",
                "install",
                "--user",
                f"--name={kernel_name}",
                f"--display-name={display_name}",
            ]

            result = self.run_command(cmd, check=False)

            if result.returncode == 0:
                success(f"✓ Kernel '{kernel_name}' registered successfully")

                # Get the exact environment name for user instructions
                python_version = self._get_python_version()
                env_name = f".venv (Python {python_version})"

                success("📓 Notebook setup complete!")
                success(f"  → In Cursor/VS Code, select '{env_name}' as your Python environment")
                success("  → Run 'metta status --components=notebook' to verify setup")
            else:
                error(f"Failed to register kernel: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)

        except Exception as e:
            error(f"Failed to install notebook support: {e}")
            raise

    def _get_python_version(self) -> str:
        """
        Get the Python version from our uv environment.

        First tries to get the actual version from the environment,
        then falls back to reading from pyproject.toml.

        Returns:
            Python version string (e.g., "3.11.7")
        """
        try:
            # Try to get version from actual environment
            result = self.run_command(
                ["uv", "run", "python", "--version"],
                check=False,
            )
            if result.returncode == 0:
                # Extract version from "Python 3.11.7" output
                version = result.stdout.strip().split()[-1]
                return version
        except Exception:
            pass

        # Fallback: read from pyproject.toml
        try:
            import tomllib

            with open("pyproject.toml", "rb") as f:
                config = tomllib.load(f)
            requires_python = config["project"]["requires-python"]
            # Extract version from "==3.11.7" format
            version = requires_python.replace("==", "")
            return version
        except Exception:
            pass

        # Last resort fallback
        return "3.11.7"

    def check_connected_as(self) -> str | None:
        """
        Check if the kernel is properly registered.

        This method uses Jupyter discovery to find our kernel and returns
        a status string if found. This is used by 'metta status' to show
        the connection status in the "Connected As" column.

        Returns:
            ".venv (Python X.X.X)" if kernel is registered, None otherwise
        """
        try:
            # Use Jupyter discovery to find our kernel
            kernel_path = self._get_kernel_path(self.KERNEL_NAME)
            if kernel_path and kernel_path.exists():
                # Get Python version for the exact environment name
                python_version = self._get_python_version()
                return f".venv (Python {python_version})"
            return None
        except Exception:
            return None
