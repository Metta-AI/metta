"""ExperimentTool - main user-facing tool for managing experiments."""

import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from metta.common.tool import Tool
from metta.experiment.launcher import ExperimentLauncher
from metta.experiment.manager import ExperimentManager
from metta.experiment.monitor import ExperimentMonitor
from metta.jobs.models import JobSpec
from metta.rl.system_config import SystemConfig


class ExperimentTool(Tool):
    """Tool for managing groups of related training/evaluation jobs.

    An experiment consists of 2-20 related jobs (e.g., parameter variations,
    A/B tests). Each experiment instance gets a unique ID and maintains state
    across job launches, monitoring, and cancellation.

    Usage modes:
    - launch (default): Start new experiment instance
    - attach: Stream logs from first running job
    - cancel: Cancel all jobs in instance
    - monitor: Show live status updates
    - report: Generate post-hoc analysis (TODO: v2)

    Example:
        def my_experiment() -> ExperimentTool:
            return ExperimentTool(
                name="lr_comparison",
                jobs=[
                    JobSpec(
                        name="lr_0001",
                        module="experiments.recipes.arena.train",
                        args={"run": "lr_comparison.lr_0001"},
                        overrides={"trainer.optimizer.learning_rate": 0.0001},
                    ),
                    JobSpec(
                        name="lr_0003",
                        module="experiments.recipes.arena.train",
                        args={"run": "lr_comparison.lr_0003"},
                        overrides={"trainer.optimizer.learning_rate": 0.0003},
                    ),
                ],
            )

        # Launch: ./tools/run.py experiments.user.my_module.my_experiment
        # Monitor: ./tools/run.py experiments.user.my_module.my_experiment mode=monitor
        # Cancel: ./tools/run.py experiments.user.my_module.my_experiment mode=cancel
    """

    # Required fields
    name: str = Field(description="Experiment name (used to generate instance ID)")
    jobs: list[JobSpec] = Field(description="List of jobs to launch")

    # Mode control (mutually exclusive)
    mode: Literal["launch", "attach", "cancel", "monitor", "report"] = Field(
        default="launch", description="Operation mode"
    )

    # Instance selection
    instance_id: Optional[str] = Field(
        default=None,
        description="Specific instance ID. If None: auto-generate (launch) or use latest (other modes)",
    )

    # Launch options
    dry_run: bool = Field(default=False, description="Show what would be launched without actually launching")
    sequential: bool = Field(
        default=False, description="Launch jobs one at a time (default: parallel). Note: v1 not implemented"
    )

    # Monitor options
    refresh_interval: int = Field(default=10, description="Seconds between status updates in monitor mode")

    # System config
    system: SystemConfig = Field(default_factory=SystemConfig)

    def invoke(self, args: dict[str, str]) -> int:
        """Main entry point from tool runner.

        Args:
            args: Additional arguments from CLI (currently unused)

        Returns:
            0 on success, non-zero on failure
        """
        # Route to appropriate handler based on mode
        if self.mode == "launch":
            return self._launch()
        elif self.mode == "attach":
            return self._attach()
        elif self.mode == "cancel":
            return self._cancel()
        elif self.mode == "monitor":
            return self._monitor()
        elif self.mode == "report":
            return self._report()
        else:
            print(f"Unknown mode: {self.mode}")
            return 1

    def _launch(self) -> int:
        """Launch new experiment instance."""
        # Generate instance ID if not provided
        if self.instance_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.instance_id = f"{self.name}_{timestamp}"

        # Get full recipe path (module + class)
        recipe = f"{self.__class__.__module__}.{self.__class__.__name__}"

        launcher = ExperimentLauncher(
            instance_id=self.instance_id,
            recipe=recipe,
            jobs=self.jobs,
            dry_run=self.dry_run,
            sequential=self.sequential,
        )

        return launcher.launch()

    def _attach(self) -> int:
        """Attach to existing experiment instance and stream logs."""
        # Find instance
        instance_id = self._resolve_instance_id()
        if instance_id is None:
            print(f"No experiment instance found for {self.name}")
            return 1

        monitor = ExperimentMonitor(instance_id)
        return monitor.attach()

    def _cancel(self) -> int:
        """Cancel all jobs in experiment instance."""
        instance_id = self._resolve_instance_id()
        if instance_id is None:
            print(f"No experiment instance found for {self.name}")
            return 1

        manager = ExperimentManager(instance_id)
        return manager.cancel_all()

    def _monitor(self) -> int:
        """Show live monitoring of experiment."""
        instance_id = self._resolve_instance_id()
        if instance_id is None:
            print(f"No experiment instance found for {self.name}")
            return 1

        monitor = ExperimentMonitor(
            instance_id,
            refresh_interval=self.refresh_interval,
        )
        return monitor.run()

    def _report(self) -> int:
        """Generate post-hoc analysis report (TODO: v2)."""
        print("Report generation not yet implemented (planned for v2)")
        print("For now, use WandB group filtering or manual notebook analysis")
        return 1

    def _resolve_instance_id(self) -> Optional[str]:
        """Resolve instance ID - use provided or find most recent.

        Returns:
            Instance ID string, or None if not found
        """
        if self.instance_id:
            return self.instance_id

        # Find most recent instance for this experiment name
        state_dir = Path("experiments/state")
        if not state_dir.exists():
            return None

        # Pattern: {name}_{timestamp}.json
        pattern = re.compile(rf"^{re.escape(self.name)}_(\d{{8}}_\d{{6}})\.json$")

        matches = []
        for path in state_dir.glob("*.json"):
            match = pattern.match(path.name)
            if match:
                timestamp = match.group(1)
                matches.append((timestamp, path.stem))

        if not matches:
            return None

        # Return most recent
        matches.sort(reverse=True)
        return matches[0][1]
