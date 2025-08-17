import logging

from metta.common.util.tool import Tool
from metta.map.load_random import get_random_map_uri
from metta.map.tools.gen import uri_is_file
from metta.map.utils.show import ShowMode, show_map
from metta.map.utils.storable_map import StorableMap

logger = logging.getLogger(__name__)


class ViewTool(Tool):
    uri: str
    mode: ShowMode

    def invoke(self):
        uri = self.uri

        if not uri_is_file(uri):
            # probably a directory
            while uri.endswith("/"):
                uri = uri[:-1]
            logger.info(f"Loading random map from directory {uri}")
            uri = get_random_map_uri(uri)

        logger.info(f"Loading map from {uri}")
        storable_map = StorableMap.from_uri(uri)

        show_map(storable_map, self.mode)
