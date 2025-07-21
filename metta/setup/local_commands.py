import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import wandb

from metta.setup.tools.local.load_policies import get_recent_runs, post_policies_to_stats, print_runs_with_artifacts
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

    def load_policies(self, unknown_args) -> None:
        """Load W&B artifacts as policies into stats database."""
        # Create parser for load-policies specific arguments
        parser = argparse.ArgumentParser(
            prog="metta local load-policies", description="Load W&B artifacts as policies into stats database"
        )
        parser.add_argument("--entity", help="W&B entity name (default: from W&B auth)")
        parser.add_argument("--project", help="W&B project name (default: 'metta')")
        parser.add_argument("--days-back", type=int, default=30, help="Number of days to look back (default: 30)")
        parser.add_argument("--limit", type=int, help="Maximum number of runs to fetch")
        parser.add_argument("--run-name", help="Specific run name to fetch (ignores days-back and limit)")
        parser.add_argument(
            "--post-policies", action="store_true", help="Post model artifacts as policies to stats database"
        )
        parser.add_argument("--stats-db-uri", help="Stats database URI (required when using --post-policies)")
        parser.add_argument("--debug", action="store_true", help="Show debug information")

        # Handle help manually since metta intercepts -h
        if "--help" in unknown_args or "-h" in unknown_args:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(unknown_args)

        # Validate that stats-db-uri is provided when post-policies is used
        if args.post_policies and not args.stats_db_uri:
            parser.error("--stats-db-uri is required when using --post-policies")

        # Get entity from args or W&B default
        api = wandb.Api()
        if args.entity:
            entity = args.entity
        else:
            entity = api.default_entity
            if not entity:
                error("No W&B entity found. Please login with 'wandb login'")
                sys.exit(1)

        # Use provided project or default to 'metta'
        project = args.project if args.project else "metta"

        info(f"Using entity: {entity}, project: {project}")

        try:
            runs = get_recent_runs(
                entity=entity,
                project=project,
                days_back=args.days_back,
                limit=args.limit,
                run_name=args.run_name,
                debug=args.debug
            )

            # Always print human-readable output
            print_runs_with_artifacts(runs, args.run_name)

            # Post policies if requested
            if args.post_policies:
                post_policies_to_stats(runs, args.stats_db_uri)

        except Exception as e:
            error(f"Error: {e}")
            sys.exit(1)
