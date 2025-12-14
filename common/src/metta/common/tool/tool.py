from __future__ import annotations

import logging
import re
from abc import abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class Tool(Config):
    """Base class for tools.

    To make a tool, you need to:
    1) Define a class that inherits from Tool (e.g., TrainTool, EvaluateTool).
    2) Define an `invoke` method that returns an exit code, optionally.
    3) Make a tool maker (function that returns an instance of your tool class).
    4) Run the tool with `./tools/run.py <path.to.tool.maker>`.

    The tool type is automatically inferred from the class name by lowercasing
    and removing the "Tool" suffix (e.g., TrainTool -> "train", EvaluateTool -> "evaluate").

    The tool maker can optionally take arguments.
    """

    system: SystemConfig = Field(default_factory=SystemConfig)

    @classmethod
    def tool_type_name(cls) -> str:
        """Infer tool type from class name.

        Tool type identifies the kind of tool (e.g., "train", "evaluate", "eval_remote").
        Removes "Tool" suffix and converts CamelCase to snake_case.
        """
        # Remove "Tool" suffix
        class_name = cls.__name__
        if class_name.endswith("Tool"):
            class_name = class_name[:-4]

        # Convert CamelCase to snake_case
        snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        return snake_case

    @abstractmethod
    def invoke(self, args: dict[str, str]) -> int | None: ...

    def output_references(self, job_name: str) -> dict:
        """
        To be consumed by stable release runner. Reflects information additional Tools can use on
        to reference this Tool's output if this job is run with run=<job_name>.
        """
        return {}


class ToolResult(BaseModel):
    result: Literal["success", "failure"]
    warnings: list[str] = []
    error: str | None = None
    output_uri: str | None = None


class ToolWithResult(Tool):
    result_file_path: str

    @abstractmethod
    def run_job(self) -> ToolResult: ...

    def invoke(self, args: dict[str, str]) -> int | None:
        try:
            result = self.run_job()
        except Exception as e:
            logger.error(f"Error running job: {e}", exc_info=True)
            result = ToolResult(result="failure", error=str(e))
        with open(self.result_file_path, "w") as f:
            f.write(result.model_dump_json())
        return 0
