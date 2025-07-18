import shutil
import subprocess
import sys
from pathlib import Path

from metta.setup.utils import error, info, success


class LocalCommands:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def build_app_backend_img(self) -> None:
        """Build local development Docker image."""
        docker_dir = self.repo_root / "app_backend"
        dockerfile_path = docker_dir / "Dockerfile"
        subprocess.run(
            ["docker", "build", "-t", "metta-app-backend:latest", "-f", str(dockerfile_path), str(self.repo_root)],
            check=True,
        )

    def build_docker_img(self, args) -> None:
        """Build local development Docker image."""
        docker_dir = self.repo_root / "devops" / "docker"
        dockerfile_path = docker_dir / "Dockerfile.local"

        if not dockerfile_path.exists():
            error(f"Dockerfile not found at {dockerfile_path}")
            sys.exit(1)

        info("Building local development Docker image...")
        info("Note: This will copy the entire repo and run install.sh during build.")
        info("This may take several minutes...")
        info("")

        # Track if we copied .metta
        copied_metta = False
        metta_home_dir = Path.home() / ".metta"
        metta_repo_dir = self.repo_root / ".metta"

        try:
            # Copy .metta directory if it exists
            if metta_home_dir.exists():
                info("Found ~/.metta directory - copying to build context")
                shutil.copytree(metta_home_dir, metta_repo_dir, dirs_exist_ok=True)
                copied_metta = True

            tag = "metta-local:latest"
            # Build the image with repo root as the build context
            cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(self.repo_root)]

            result = subprocess.run(cmd, cwd=self.repo_root)

            if result.returncode == 0:
                info("")
                info("Note: The container has a full copy of the repo at build time.")
                info("Local changes won't be reflected unless you rebuild or attach.")
                success(f"Build complete! Image available as {tag}")
            else:
                error("Build failed!")
                sys.exit(result.returncode)

        finally:
            # Clean up .metta directory if we copied it
            if copied_metta and metta_repo_dir.exists():
                info("Cleaning up .metta directory from build context")
                shutil.rmtree(metta_repo_dir)
