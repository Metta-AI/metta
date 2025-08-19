import logging

from metta.common.config.tool import Tool
from metta.map.utils.show import ShowMode, show_map
from metta.map.utils.storable_map import StorableMap

logger = logging.getLogger(__name__)


class ViewTool(Tool):
    uri: str
    mode: ShowMode

    def invoke(self):
        uri = self.uri

        logger.info(f"Loading map from {uri}")
        storable_map = StorableMap.from_uri(uri)

        show_map(storable_map, self.mode)
