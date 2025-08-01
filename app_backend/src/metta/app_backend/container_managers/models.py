from pydantic import BaseModel

from metta.app_backend.routes.eval_task_routes import TaskResponse


class WorkerInfo(BaseModel):
    container_id: str
    container_name: str
    git_hashes: list[str] = []
    assigned_task: TaskResponse | None = None

    def __str__(self) -> str:
        git_hash_str = f"hashes={','.join(self.git_hashes)}"
        return f"WorkerInfo({git_hash_str}, id={self.container_id[:3]}, name={self.container_name[:3]})"

    def __repr__(self) -> str:
        return self.__str__()
