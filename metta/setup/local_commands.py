#!/usr/bin/env -S uv run
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.fs import get_repo_root
from metta.rl.checkpoint_manager import CheckpointManager, is_valid_checkpoint_filename
from metta.setup.utils import error, info
from metta.sim.utils import get_or_create_policy_ids

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
        load_parser = subparsers.add_parser("load-policies", help="Register local checkpoints with stats server")
        load_parser.add_argument(
            "--data-dir", default="./train_dir", help="Training data directory (default: ./train_dir)"
        )
        load_parser.add_argument("--stats-db-uri", required=True, help="Stats database URI")
        load_parser.add_argument("--run-name", help="Specific run name to load (loads all if not specified)")

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
            self.load_policies(args)
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
            from metta.setup.tools.local.kind import KindLocal

            self._kind_manager = KindLocal()
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

    def load_policies(self, args) -> None:
        """Load local checkpoint directories as policies into stats database."""
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            error(f"Data directory does not exist: {args.data_dir}")
            sys.exit(1)

        info(f"Scanning for checkpoints in: {args.data_dir}")

        checkpoint_tuples = []
        for run_dir in data_dir.iterdir():
            if not run_dir.is_dir() or (args.run_name and run_dir.name != args.run_name):
                continue

            checkpoint_subdir = run_dir / "checkpoints"
            if checkpoint_subdir.exists():
                valid_checkpoints = []
                for ckpt in checkpoint_subdir.glob("*.pt"):
                    if is_valid_checkpoint_filename(ckpt.name):
                        uri = CheckpointManager.normalize_uri(str(ckpt))
                        metadata = CheckpointManager.get_policy_metadata(uri)
                        valid_checkpoints.append((ckpt, metadata["epoch"]))

                if valid_checkpoints:
                    latest_ckpt = max(valid_checkpoints, key=lambda x: x[1])[0]
                    checkpoint_tuples.append(
                        (run_dir.name, f"file://{latest_ckpt}", f"Local checkpoint from {run_dir.name}")
                    )

        if not checkpoint_tuples:
            info("No valid checkpoints found")
            return

        info(f"Found {len(checkpoint_tuples)} checkpoint directories")

        stats_client = StatsClient.create(args.stats_db_uri)
        if not stats_client:
            error("Failed to connect to stats database")
            sys.exit(1)

        policy_ids = get_or_create_policy_ids(stats_client, checkpoint_tuples)
        json_repr = json.dumps({name: str(pid) for name, pid in policy_ids.items()}, indent=2)
        info(f"Registered {len(policy_ids)} policies:")
        print(json_repr)

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
