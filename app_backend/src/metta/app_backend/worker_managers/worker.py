import typing

import pydantic

# Kubernetes pod phases plus additional statuses for other container systems
WorkerStatus = typing.Literal["Pending", "Running", "Succeeded", "Failed", "Unknown"]


class Worker(pydantic.BaseModel):
    """Represents a worker with its name and current status."""

    name: str
    status: WorkerStatus
