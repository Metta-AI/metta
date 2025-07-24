import argparse
import subprocess
import sys
from pathlib import Path

import wandb

from metta.common.util.fs import get_repo_root
from metta.setup.tools.local.kind import Kind
from metta.setup.tools.local.load_policies import get_recent_runs, post_policies_to_stats, print_runs_with_artifacts
from metta.setup.utils import error, info


class LocalCommands:
    def __init__(self):
        self.repo_root = get_repo_root()
        self._kind_manager = Kind()

    def _build_img(self, tag: str, dockerfile_path: Path, build_args: list[str] | None = None) -> None:
        cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path)]
        if build_args:
            cmd.extend(build_args)
        cmd.append(str(self.repo_root))
        subprocess.run(cmd, check=True)

    def build_app_backend_img(self) -> None:
        self._build_img("metta-app-backend:latest", self.repo_root / "app_backend" / "Dockerfile")

    def build_policy_evaluator_img(
        self, tag: str = "metta-policy-evaluator-local:latest", build_args: list[str] | None = None
    ) -> None:
        self._build_img(
            tag,
            self.repo_root / "devops" / "docker" / "Dockerfile.policy_evaluator",
            build_args or [],
        )

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
                debug=args.debug,
            )

            # Always print human-readable output
            print_runs_with_artifacts(runs, args.run_name)

            # Post policies if requested
            if args.post_policies:
                post_policies_to_stats(runs, args.stats_db_uri)

        except Exception as e:
            error(f"Error: {e}")
            sys.exit(1)

    def kind(self, args) -> None:
        """Handle Kind cluster management for Kubernetes testing."""
        action = args.action

        if action == "build":
            self._kind_manager.build()
        elif action == "up":
            self._kind_manager.up()
        elif action == "down":
            self._kind_manager.down()
        elif action == "clean":
            self._kind_manager.clean()
        elif action == "get-pods":
            self._kind_manager.get_pods()
        elif action == "logs":
            if hasattr(args, "pod_name") and args.pod_name:
                self._kind_manager.logs(args.pod_name)
            else:
                error("Pod name is required for logs command")
                sys.exit(1)
        elif action == "enter":
            if hasattr(args, "pod_name") and args.pod_name:
                self._kind_manager.enter(args.pod_name)
            else:
                error("Pod name is required for enter command")
                sys.exit(1)
