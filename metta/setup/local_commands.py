import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import wandb
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.common.util.fs import get_repo_root
from metta.common.util.stats_client_cfg import get_stats_client_direct
from metta.common.wandb.wandb_runs import find_training_runs
from metta.setup.tools.local.kind import Kind
from metta.setup.utils import error, info
from metta.sim.utils import get_or_create_policy_ids


class LocalCommands:
    def __init__(self):
        self.repo_root = get_repo_root()
        self._kind_manager = Kind(self.repo_root)

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

        # Use provided project or default to 'metta'
        project = args.project if args.project else "metta"

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
