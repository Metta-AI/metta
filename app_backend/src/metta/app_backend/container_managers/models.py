from pydantic import BaseModel


class WorkerInfo(BaseModel):
    git_hash: str
    container_id: str
    container_name: str

    def __str__(self) -> str:
        return f"WorkerInfo(hash={self.git_hash[:3]}, id={self.container_id[:3]}, name={self.container_name[:3]})"

    def __repr__(self) -> str:
        return self.__str__()
