"""Routing dispatcher for hybrid execution modes."""

import logging
from typing import Optional

from metta.sweep.models import JobDefinition, JobTypes
from metta.sweep.protocols import Dispatcher

logger = logging.getLogger(__name__)


class RoutingDispatcher(Dispatcher):
    """Routes jobs to different dispatchers based on job type."""

    def __init__(
        self,
        routes: dict[JobTypes, Dispatcher],
        default_dispatcher: Optional[Dispatcher] = None,
    ):
        """Initialize with job type to dispatcher mapping."""
        self.routes = routes
        self.default_dispatcher = default_dispatcher

    def dispatch(self, job: JobDefinition) -> str:
        """Route job to appropriate dispatcher."""
        dispatcher = self.routes.get(job.type, self.default_dispatcher)

        if dispatcher is None:
            raise ValueError(
                f"No dispatcher configured for job type {job.type.value}. Available routes: {list(self.routes.keys())}"
            )

        return dispatcher.dispatch(job)
