from mettagrid.mapgen.scene import Scene, SceneConfig


class NopConfig(SceneConfig):
    pass


class Nop(Scene[NopConfig]):
    """
    This scene doesn't do anything.
    """

    def render(self):
        pass
