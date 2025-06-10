import logging
import os
import uuid

import numpy as np

from metta.map.scene import Scene, make_scene
from metta.map.types import SceneCfg
from metta.util.config import Config

logger = logging.getLogger(__name__)


class SceneCacheParams(Config):
    cache_dir: str | None = None
    cache_size: int
    scene: SceneCfg


class SceneCache(Scene[SceneCacheParams]):
    def post_init(self):
        # random dir in /tmp
        self.cache_dir = self.params.cache_dir or os.path.join("/tmp", str(uuid.uuid4()))
        os.makedirs(self.cache_dir, exist_ok=True)

    def render(self):
        # Note: this approach won't allow us to introspect the scene tree in the future tools.
        # We'll need to rethink it when we get there.
        scene_id = self.rng.integers(1, self.params.cache_size + 1)
        scene_path = os.path.join(self.cache_dir, f"{scene_id}.npy")
        if os.path.exists(scene_path):
            logger.info("Loading scene from %s", scene_path)
            loaded_grid = np.load(scene_path)
            if loaded_grid.shape != (self.width, self.height):
                raise ValueError(f"Loaded grid has shape {loaded_grid.shape}, expected {self.width, self.height}")
            self.grid[:] = loaded_grid
        else:
            scene = make_scene(self.params.scene, self.grid)
            scene.render_with_children()

            logger.info("Saving scene to %s", scene_path)
            np.save(scene_path, self.grid)
