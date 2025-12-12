from pydantic import Field

from metta.common.tool import Tool
from mettagrid.util.uri_resolvers.schemes import resolve_uri


class ResolveUriTool(Tool):
    uri: str = Field(description="The URI to resolve")

    def invoke(self, args: dict[str, str]) -> int | None:
        print(resolve_uri(self.uri).canonical)
        return 0
