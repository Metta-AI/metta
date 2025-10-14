from abc import abstractmethod
from typing import Literal

from pydantic import BaseModel

from metta.common.tool.tool import Tool


class JobResult(BaseModel):
    result: Literal["success", "failure"]
    warnings: list[str] = []
    error: str | None = None
    output_uri: str | None = None


class RemoteJobTool(Tool):
    output_file_path: str

    @abstractmethod
    def run_job(self) -> JobResult: ...

    def invoke(self, args: dict[str, str]) -> int | None:
        result = self.run_job()
        with open(self.output_file_path, "w") as f:
            f.write(result.model_dump_json())
        return 0
