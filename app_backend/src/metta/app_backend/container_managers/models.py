from typing import Optional

from pydantic import BaseModel

from metta.app_backend.routes.eval_task_routes import TaskResponse


class WorkerInfo(BaseModel):
    git_hash: str
    container_id: str
    container_name: str
    alive: bool
    task: Optional[TaskResponse] = None

    def __str__(self) -> str:
        return (
            f"WorkerInfo(hash={self.git_hash[:3]}, id={self.container_id[:3]}, "
            f"name={self.container_name[:3]}, alive={self.alive}, task={str(self.task.id)[:3] if self.task else None})"
        )

    def __repr__(self) -> str:
        return self.__str__()
