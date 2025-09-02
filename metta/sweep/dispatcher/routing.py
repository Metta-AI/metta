"""Routing dispatcher that routes jobs to different dispatchers based on criteria."""

import logging
from typing import Optional

from metta.sweep.models import JobDefinition, JobTypes
from metta.sweep.protocols import Dispatcher

logger = logging.getLogger(__name__)


class RoutingDispatcher:
    """Routes jobs to different dispatchers based on job type for hybrid execution."""

    def __init__(
        self,
        routes: dict[JobTypes, Dispatcher],
        default_dispatcher: Optional[Dispatcher] = None,
    ):
        """Initialize with job type to dispatcher mapping and optional default."""
        self.routes = routes
        self.default_dispatcher = default_dispatcher

    def dispatch(self, job: JobDefinition) -> str:
        """Route job to appropriate dispatcher and return dispatch_id."""
        dispatcher = self.routes.get(job.type, self.default_dispatcher)

        if dispatcher is None:
            raise ValueError(
                f"No dispatcher configured for job type {job.type.value}. Available routes: {list(self.routes.keys())}"
            )

        logger.debug(f"Routing {job.type.value} to {type(dispatcher).__name__}")
        return dispatcher.dispatch(job)

    def get_dispatcher_for_job_type(self, job_type: JobTypes) -> Optional[Dispatcher]:
        """Get dispatcher for specific job type, useful for introspection."""
        return self.routes.get(job_type, self.default_dispatcher)
