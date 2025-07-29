#!/usr/bin/env -S uv run
import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from metta.common.util.constants import METTA_WANDB_PROJECT
from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info

# Type checking imports
if TYPE_CHECKING:
    pass


def setup_local_parser(parser: argparse.ArgumentParser) -> None:
    """Setup local subcommands parser for compatibility with metta CLI.

    This just adds a simple help message since the actual parsing
    is delegated to LocalCommands.main().
    """
    # No arguments needed - we'll get unknown_args in the handler
    # Store parser for help display
    parser.set_defaults(local_parser=parser)


class LocalCommands:
    def __init__(self):
        self.repo_root = get_repo_root()
        self._kind_manager = None

    def _build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser for local commands."""
        parser = argparse.ArgumentParser(
            description="Metta Local Development Commands",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Build commands
        subparsers.add_parser("build-policy-evaluator-img", help="Build policy evaluator Docker image")
        subparsers.add_parser("build-app-backend-img", help="Build app backend Docker image")

        # Load policies command
        load_parser = subparsers.add_parser("load-policies", help="Load W&B artifacts as policies")
        load_parser.add_argument("--entity", help="W&B entity name (default: from W&B auth)")
        load_parser.add_argument("--project", help=f"W&B project name (default: '{METTA_WANDB_PROJECT}')")
        load_parser.add_argument("--days-back", type=int, default=30, help="Number of days to look back (default: 30)")
        load_parser.add_argument("--limit", type=int, help="Maximum number of runs to fetch")
        load_parser.add_argument("--run-name", help="Specific run name to fetch (ignores days-back and limit)")
        load_parser.add_argument("--stats-db-uri", help="Stats database URI (required when using --post-policies)")

        # Server commands
        subparsers.add_parser("stats-server", help="Launch Stats Server")
        subparsers.add_parser("observatory", help="Launch Observatory")

        # Kind command
        kind_parser = subparsers.add_parser("kind", help="Manage Kind cluster")
        kind_subparsers = kind_parser.add_subparsers(dest="action", help="Kind actions")
        kind_subparsers.add_parser("build", help="Create Kind cluster")
        kind_subparsers.add_parser("up", help="Start orchestrator")
        kind_subparsers.add_parser("down", help="Stop orchestrator")
        kind_subparsers.add_parser("clean", help="Delete cluster")
        kind_subparsers.add_parser("get-pods", help="List pods")

        logs_parser = kind_subparsers.add_parser("logs", help="Follow pod logs")
        logs_parser.add_argument("pod_name", help="Pod name")

        enter_parser = kind_subparsers.add_parser("enter", help="Enter pod shell")
        enter_parser.add_argument("pod_name", help="Pod name")

        return parser

    def main(self, argv=None) -> None:
        """Main entry point for local commands CLI."""
        parser = self._build_parser()

        # Parse arguments
        if argv is None:
            argv = []

        # Always use parse_known_args for consistency
        args, unknown_args = parser.parse_known_args(argv)

        # Dispatch to command handler
        if not args.command:
            parser.print_help()
            return

        if args.command == "build-policy-evaluator-img":
            self.build_policy_evaluator_img(build_args=unknown_args)
        elif args.command == "build-app-backend-img":
            self.build_app_backend_img()
        elif args.command == "load-policies":
            # Convert back to list format for compatibility
            load_args = []
            if args.entity:
                load_args.extend(["--entity", args.entity])
            if args.project:
                load_args.extend(["--project", args.project])
            if args.days_back != 30:
                load_args.extend(["--days-back", str(args.days_back)])
            if args.limit:
                load_args.extend(["--limit", str(args.limit)])
            if args.run_name:
                load_args.extend(["--run-name", args.run_name])
            if args.stats_db_uri:
                load_args.extend(["--stats-db-uri", args.stats_db_uri])
            self.load_policies(load_args)
        elif args.command == "kind":
            self.kind(args)
        elif args.command == "observatory":
            self.observatory(args, unknown_args)
        elif args.command == "stats-server":
            self.stats_server(args, unknown_args)
        else:
            error(f"Unknown command: {args.command}")
            sys.exit(1)

    @property
    def kind_manager(self):
        if self._kind_manager is None:
            from metta.setup.tools.local.kind import Kind

            self._kind_manager = Kind()
        return self._kind_manager

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
        # Lazy imports
        import wandb
        from omegaconf import DictConfig

        from metta.agent.policy_store import PolicyStore
        from metta.common.util.constants import METTA_WANDB_PROJECT
        from metta.common.util.stats_client_cfg import get_stats_client_direct
        from metta.common.wandb.wandb_runs import find_training_runs
        from metta.sim.utils import get_or_create_policy_ids

        # Create parser for load-policies specific arguments
        parser = argparse.ArgumentParser(
            prog="metta local load-policies", description="Load W&B artifacts as policies into stats database"
        )
        parser.add_argument("--entity", help="W&B entity name (default: from W&B auth)")
        parser.add_argument("--project", help=f"W&B project name (default: '{METTA_WANDB_PROJECT}')")
        parser.add_argument("--days-back", type=int, default=30, help="Number of days to look back (default: 30)")
        parser.add_argument("--limit", type=int, help="Maximum number of runs to fetch")
        parser.add_argument("--run-name", help="Specific run name to fetch (ignores days-back and limit)")
        parser.add_argument("--stats-db-uri", help="Stats database URI (required when using --post-policies)")

        # Handle help manually since metta intercepts -h
        if "--help" in unknown_args or "-h" in unknown_args:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(unknown_args)

        # Get entity from args or W&B default
        api = wandb.Api()
        if args.entity:
            entity = args.entity
        else:
            entity = api.default_entity
            if not entity:
                error("No W&B entity found. Please login with 'wandb login'")
                sys.exit(1)

        # Use provided project or default to METTA_WANDB_PROJECT
        project = args.project if args.project else METTA_WANDB_PROJECT

        info(f"Using entity: {entity}, project: {project}")
        if not args.stats_db_uri:
            print("\nNo STATS_DB_URI provided, skipping policy posting.")
            return

        print(f"\nConnecting to stats database at {args.stats_db_uri}...")
        logger = logging.getLogger(__name__)
        stats_client = get_stats_client_direct(args.stats_db_uri, logger)
        if not stats_client:
            print("No stats client")
            return
        stats_client.validate_authenticated()
        runs = find_training_runs(
            entity=entity,
            project=project,
            created_after=(datetime.now() - timedelta(days=args.days_back)).isoformat(),
            limit=args.limit,
            run_names=[args.run_name] if args.run_name else None,
        )
        policy_store = PolicyStore(
            DictConfig(
                dict(
                    wandb=dict(
                        enabled=True,
                        project=project,
                        entity=entity,
                    ),
                    device="cpu",
                )
            ),
            wandb_run=None,
        )
        policy_records = []
        for run in runs:
            uri = f"wandb://run/{run.name}"
            # n and metric are ignored
            policy_records.extend(policy_store.policy_records(uri, selector_type="all", n=1, metric="top"))
        policy_ids = get_or_create_policy_ids(
            stats_client,
            [(pr.run_name, pr.uri, None) for pr in policy_records],
        )
        json_repr = json.dumps({name: str(pid) for name, pid in policy_ids.items()}, indent=2)
        print(f"Ensured {len(policy_ids)} policy IDs: {json_repr}")
        sys.stdout.flush()
        sys.stderr.flush()

    def kind(self, args) -> None:
        """Handle Kind cluster management for Kubernetes testing."""
        action = args.action

        if action == "build":
            self.kind_manager.build()
        elif action == "up":
            self.kind_manager.up()
        elif action == "down":
            self.kind_manager.down()
        elif action == "clean":
            self.kind_manager.clean()
        elif action == "get-pods":
            self.kind_manager.get_pods()
        elif action == "logs":
            if hasattr(args, "pod_name") and args.pod_name:
                self.kind_manager.logs(args.pod_name)
            else:
                error("Pod name is required for logs command")
                sys.exit(1)
        elif action == "enter":
            if hasattr(args, "pod_name") and args.pod_name:
                self.kind_manager.enter(args.pod_name)
            else:
                error("Pod name is required for enter command")
                sys.exit(1)

    def observatory(self, args, unknown_args=None) -> None:
        """Launch Observatory with specified backend."""
        # Build the command to run launch.py
        cmd = [sys.executable, str(self.repo_root / "observatory" / "launch.py")]

        # Pass through any arguments
        if unknown_args:
            cmd.extend(unknown_args)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            error(f"Failed to launch Observatory: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            info("\nObservatory shutdown")
            sys.exit(0)

    def stats_server(self, args, unknown_args=None) -> None:
        """Launch Stats Server."""
        cmd = [
            "uv",
            "run",
            "python",
            str(self.repo_root / "app_backend" / "src" / "metta" / "app_backend" / "server.py"),
            *(unknown_args or []),
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            error(f"Failed to launch Stats Server: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            info("\nStats Server shutdown")
            sys.exit(0)


def main():
    """Entry point for standalone execution."""
    local_commands = LocalCommands()
    local_commands.main(sys.argv[1:])


if __name__ == "__main__":
    main()
