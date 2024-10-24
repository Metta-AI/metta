
from libc.stdio cimport printf


import numpy as np
cimport numpy as cnp
import gymnasium as gym
from omegaconf import OmegaConf
from types import SimpleNamespace

from puffergrid.grid_env cimport GridEnv
from puffergrid.grid_object cimport GridObject
from puffergrid.observation_encoder cimport ObsType

from mettagrid.objects cimport ObjectLayers, Agent, ResetHandler, Wall, Generator, Converter, Altar
from mettagrid.observation_encoder cimport MettaObservationEncoder, MettaCompactObservationEncoder
from mettagrid.actions.move import Move
from mettagrid.actions.rotate import Rotate
from mettagrid.actions.use import Use
from mettagrid.actions.attack import Attack
from mettagrid.actions.shield import Shield
from mettagrid.actions.gift import Gift
from mettagrid.actions.noop import Noop
from mettagrid.actions.swap import Swap

obs_np_type = np.uint8

cdef class MettaGrid(GridEnv):
    cdef:
        object _cfg

    def __init__(self, env_cfg: OmegaConf, map: np.ndarray):
        self._cfg = OmegaConf.create(env_cfg.game)
        cfg = self._cfg

        obs_encoder = MettaObservationEncoder()
        if env_cfg.compact_obs:
            obs_encoder = MettaCompactObservationEncoder()

        actions = []
        if cfg.actions.noop.enabled:
            actions.append(Noop(cfg.actions.noop))
        if cfg.actions.move.enabled:
            actions.append(Move(cfg.actions.move))
        if cfg.actions.rotate.enabled:
            actions.append(Rotate(cfg.actions.rotate))
        if cfg.actions.use.enabled:
            actions.append(Use(cfg.actions.use))
        if cfg.actions.attack.enabled:
            actions.append(Attack(cfg.actions.attack))
        if cfg.actions.shield.enabled:
            actions.append(Shield(cfg.actions.shield))
        if cfg.actions.gift.enabled:
            actions.append(Gift(cfg.actions.gift))
        if cfg.actions.swap.enabled:
            actions.append(Swap(cfg.actions.swap))

        GridEnv.__init__(
            self,
            cfg.num_agents,
            map.shape[1],
            map.shape[0],
            cfg.max_steps,
            dict(ObjectLayers).values(),
            cfg.obs_width, cfg.obs_height,
            obs_encoder,
            actions,
            [ ResetHandler() ],
            use_flat_actions=env_cfg.flatten_actions
        )

        cdef Agent *agent
        for r in range(map.shape[0]):
            for c in range(map.shape[1]):
                if map[r,c] == "W":
                    self._grid.add_object(new Wall(r, c, cfg.objects.wall))
                    self._stats.game_incr("objects.wall")
                elif map[r,c] == "g":
                    self._grid.add_object(new Generator(r, c, cfg.objects.generator))
                    self._stats.game_incr("objects.generator")
                elif map[r,c] == "c":
                    self._grid.add_object(new Converter(r, c, cfg.objects.converter))
                    self._stats.game_incr("objects.converter")
                elif map[r,c] == "a":
                    self._grid.add_object(new Altar(r, c, cfg.objects.altar))
                    self._stats.game_incr("objects.altar")
                elif map[r,c][0] == "A":
                    agent = new Agent(r, c, cfg.objects.agent)
                    self._grid.add_object(agent)
                    self.add_agent(agent)
                    self._stats.game_incr("objects.agent")


    def render(self):
        grid = self.render_ascii(["A", "#", "g", "c", "a"])
        for r in grid:
            print("".join(r))

    cpdef grid_objects(self):
        cdef GridObject *obj
        cdef ObsType[:] obj_data = np.zeros(len(self._obs_encoder.feature_names()), dtype=obs_np_type)
        cdef unsigned int obj_id, i
        cdef MettaObservationEncoder obs_encoder = <MettaObservationEncoder>self._obs_encoder
        objects = {}
        for obj_id in range(1, self._grid.objects.size()):
            obj = self._grid.object(obj_id)
            if obj == NULL:
                continue
            objects[obj_id] = {
                "id": obj_id,
                "type": obj._type_id,
                "r": obj.location.r,
                "c": obj.location.c,
                "layer": obj.location.layer
            }
            obs_encoder._encode(obj, obj_data, 0)
            for i, name in enumerate(obs_encoder._type_feature_names[obj._type_id]):
                objects[obj_id][name] = obj_data[i]

        for agent_idx in range(self._agents.size()):
            objects[self._agents[agent_idx].id]["agent_id"] = agent_idx

        return objects
