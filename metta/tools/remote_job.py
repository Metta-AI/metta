import abc
import typing

import pydantic

import metta.common.tool.tool


class JobResult(pydantic.BaseModel):
    result: typing.Literal["success", "failure"]
    warnings: list[str] = []
    error: str | None = None
    output_uri: str | None = None


class RemoteJobTool(metta.common.tool.tool.Tool):
    job_result_file_path: str

    @abc.abstractmethod
    def run_job(self) -> JobResult: ...

    def invoke(self, args: dict[str, str]) -> int | None:
        result = self.run_job()
        with open(self.job_result_file_path, "w") as f:
            f.write(result.model_dump_json())
        return 0
