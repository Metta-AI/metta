import mettagrid.mapgen.scene


class NopConfig(mettagrid.mapgen.scene.SceneConfig):
    pass


class Nop(mettagrid.mapgen.scene.Scene[NopConfig]):
    """
    This scene doesn't do anything.
    """

    def render(self):
        pass
