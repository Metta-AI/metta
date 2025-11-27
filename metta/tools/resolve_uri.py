from pydantic import Field

from metta.common.tool import Tool
from mettagrid.util.url_schemes import resolve_uri


class ResolveUriTool(Tool):
    uri: str = Field(description="The URI to resolve")

    def invoke(self, args: dict[str, str]) -> int | None:
        print(resolve_uri(self.uri))
        return 0
