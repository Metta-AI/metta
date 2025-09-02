import logging

from metta.common.config.tool import Tool
from metta.mettagrid.mapgen.utils.show import ShowMode, show_map
from metta.mettagrid.mapgen.utils.storable_map import StorableMap

logger = logging.getLogger(__name__)


class ViewTool(Tool):
    uri: str
    mode: ShowMode

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        uri = self.uri

        logger.info(f"Loading map from {uri}")
        storable_map = StorableMap.from_uri(uri)

        show_map(storable_map, self.mode)
