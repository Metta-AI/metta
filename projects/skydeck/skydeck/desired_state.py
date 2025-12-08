"""Desired state manager for experiments."""

import logging
from typing import Optional

from .database import Database
from .models import CreateExperimentRequest, DesiredState, Experiment, JobStatus

logger = logging.getLogger(__name__)


class DesiredStateManager:
    """Manages desired state of experiments.

    This handles CRUD operations for experiments and tracks what
    the user wants to be running (desired state).
    """

    def __init__(self, db: Database):
        """Initialize desired state manager.

        Args:
            db: Database instance
        """
        self.db = db

    async def create_experiment(self, request: CreateExperimentRequest) -> Experiment:
        """Create a new experiment.

        Args:
            request: Experiment creation request

        Returns:
            Created Experiment object
        """
        # Check if experiment already exists
        existing = await self.db.get_experiment(request.id)
        if existing:
            raise ValueError(f"Experiment with id '{request.id}' already exists")

        experiment = Experiment(
            id=request.id,
            name=request.name,
            flags=request.flags,
            base_command=request.base_command,
            run_name=request.id,  # Use experiment ID as run name
            nodes=request.nodes,
            gpus=request.gpus,
            instance_type=request.instance_type,
            cloud=request.cloud,
            region=request.region,
            zone=request.zone,
            spot=request.spot,
            desired_state=request.desired_state,
            current_state=JobStatus.INIT,
            description=request.description,
            tags=request.tags,
        )

        await self.db.save_experiment(experiment)
        logger.info(f"Created experiment: {experiment.id} (desired={experiment.desired_state})")
        return experiment

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment object or None
        """
        return await self.db.get_experiment(experiment_id)

    async def get_all_experiments(self) -> list[Experiment]:
        """Get all experiments.

        Returns:
            List of all experiments
        """
        return await self.db.get_all_experiments()

    async def delete_experiment(self, experiment_id: str):
        """Delete an experiment and all its jobs.

        Args:
            experiment_id: Experiment ID to delete
        """
        experiment = await self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        await self.db.delete_experiment(experiment_id)
        logger.info(f"Deleted experiment: {experiment_id}")

    async def update_desired_state(self, experiment_id: str, desired_state: DesiredState) -> Experiment:
        """Update experiment desired state.

        Args:
            experiment_id: Experiment ID
            desired_state: New desired state

        Returns:
            Updated Experiment object
        """
        experiment = await self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        await self.db.update_experiment_desired_state(experiment_id, desired_state)
        experiment.desired_state = desired_state

        logger.info(f"Updated experiment {experiment_id} desired_state to {desired_state}")
        return experiment

    async def update_flags(self, experiment_id: str, flags: dict) -> Experiment:
        """Update experiment flags.

        Args:
            experiment_id: Experiment ID
            flags: New flags dict

        Returns:
            Updated Experiment object
        """
        experiment = await self.db.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        await self.db.update_experiment_flags(experiment_id, flags)
        experiment.flags = flags

        logger.info(f"Updated experiment {experiment_id} flags")
        return experiment

    async def get_experiments_needing_reconciliation(self) -> list[Experiment]:
        """Get all experiments that need reconciliation.

        Returns:
            List of experiments where current_state != desired_state
        """
        all_experiments = await self.db.get_all_experiments()
        return [exp for exp in all_experiments if exp.needs_reconciliation()]
