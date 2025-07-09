from metta.common.util.config import Config
from metta.map.mapgen import MapGen


class MapGenAsciiParams(Config):
    uri: str
    border_width: int = 0


class MapGenAscii(MapGen):
    """
    Shortcut for MapGen with ASCII scene.

    It allows using this syntax:

    ```yaml
    game:
      map_builder:
        _target_: metta.map.mapgen_ascii.MapGenAscii
        uri: "path/to/my/map.map"
    ```

    Instead of:

    ```yaml
    game:
      map_builder:
        _target_: metta.map.mapgen.MapGen
        border_width: 0
        root:
          type: metta.map.scenes.ascii.Ascii
          params:
            uri: "path/to/my/map.map"
    ```

    This class doesn't support most mapgen features. For example, it doesn't support `instances`.

    If you need more than just a single ASCII scene, use MapGen directly.
    """

    def __init__(self, **kwargs):
        ascii_params = MapGenAsciiParams(**kwargs)
        super().__init__(
            border_width=ascii_params.border_width,
            root={"type": "metta.map.scenes.ascii.Ascii", "params": {"uri": ascii_params.uri}},
        )
