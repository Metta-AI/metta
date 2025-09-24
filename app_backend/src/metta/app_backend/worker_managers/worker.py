from typing import Literal

from pydantic import BaseModel

# Kubernetes pod phases plus additional statuses for other container systems
WorkerStatus = Literal["Pending", "Running", "Succeeded", "Failed", "Unknown"]


class Worker(BaseModel):
    """Represents a worker with its name and current status."""

    name: str
    status: WorkerStatus
