from types import SimpleNamespace

import numpy as np
cimport numpy as cnp
import gymnasium as gym
from omegaconf import OmegaConf

# C/C++ imports
from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map

# Core mettagrid imports
from mettagrid.grid_env cimport GridEnv
from mettagrid.grid_object cimport GridObject
from mettagrid.observation_encoder cimport (
    ObsType,
    ObservationEncoder,
    SemiCompactObservationEncoder
)

# Object imports
from mettagrid.objects.mine cimport Mine
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.production_handler cimport ProductionHandler
from mettagrid.objects.wall cimport Wall
from mettagrid.objects.converter cimport Converter
from mettagrid.objects.generator cimport Generator
from mettagrid.objects.altar cimport Altar
from mettagrid.objects.lab cimport Lab
from mettagrid.objects.factory cimport Factory
from mettagrid.objects.temple cimport Temple
from mettagrid.objects.armory cimport Armory
from mettagrid.objects.lasery cimport Lasery
from mettagrid.objects.constants cimport ObjectLayers, InventoryItemNames


# Action imports
from mettagrid.actions.move import Move
from mettagrid.actions.rotate import Rotate
from mettagrid.actions.get_output import GetOutput
from mettagrid.actions.put_recipe_items import PutRecipeItems
from mettagrid.actions.attack import Attack
from mettagrid.actions.attack_nearest import AttackNearest
from mettagrid.actions.noop import Noop
from mettagrid.actions.swap import Swap
from mettagrid.actions.change_color import ChangeColorAction

cdef class MettaGrid(GridEnv):
    cdef:
        object _cfg
        map[unsigned int, float] _group_reward_pct
        map[unsigned int, unsigned int] _group_sizes
        cnp.ndarray _group_rewards_np
        double[:] _group_rewards

    def __init__(self, env_cfg: OmegaConf, map: np.ndarray):
        cfg = OmegaConf.create(env_cfg.game)
        self._cfg = cfg

        obs_encoder = ObservationEncoder()
        if env_cfg.semi_compact_obs:
            obs_encoder = SemiCompactObservationEncoder()
        actions = []
        if cfg.actions.put_items.enabled:
            actions.append(PutRecipeItems(cfg.actions.put_items))
        if cfg.actions.get_items.enabled:
            actions.append(GetOutput(cfg.actions.get_items))
        if cfg.actions.noop.enabled:
            actions.append(Noop(cfg.actions.noop))
        if cfg.actions.move.enabled:
            actions.append(Move(cfg.actions.move))
        if cfg.actions.rotate.enabled:
            actions.append(Rotate(cfg.actions.rotate))
        if cfg.actions.attack.enabled:
            actions.append(Attack(cfg.actions.attack))
            actions.append(AttackNearest(cfg.actions.attack))
        if cfg.actions.swap.enabled:
            actions.append(Swap(cfg.actions.swap))
        if cfg.actions.change_color.enabled:
            actions.append(ChangeColorAction(cfg.actions.change_color))

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
            track_last_action=env_cfg.track_last_action
        )

        self._group_rewards_np = np.zeros(len(cfg.groups))
        self._group_rewards = self._group_rewards_np
        self._group_sizes = {
            g.id: 0 for g in cfg.groups.values()
        }
        self._group_reward_pct = {
            g.id: g.get("group_reward_pct", 0) for g in cfg.groups.values()
        }

        cdef Agent *agent
        cdef Converter *converter = NULL
        cdef string group_name
        cdef unsigned char group_id
        for r in range(map.shape[0]):
            for c in range(map.shape[1]):

                if map[r,c] == "wall":
                    wall = new Wall(r, c, cfg.objects.wall)
                    self._grid.add_object(wall)
                    self._stats.incr(b"objects.wall")

                elif map[r,c] == "mine":
                    converter = new Mine(r, c, cfg.objects.mine)
                elif map[r,c] == "generator":
                    converter = new Generator(r, c, cfg.objects.generator)
                elif map[r,c] == "altar":
                    converter = new Altar(r, c, cfg.objects.altar)
                elif map[r,c] == "armory":
                    converter = new Armory(r, c, cfg.objects.armory)
                elif map[r,c] == "lasery":
                    converter = new Lasery(r, c, cfg.objects.lasery)
                elif map[r,c] == "lab":
                    converter = new Lab(r, c, cfg.objects.lab)
                elif map[r,c] == "factory":
                    converter = new Factory(r, c, cfg.objects.factory)
                elif map[r,c] == "temple":
                    converter = new Temple(r, c, cfg.objects.temple)

                elif map[r,c].startswith("agent."):
                    group_name = map[r,c].split(".")[1]
                    agent_cfg = OmegaConf.to_container(OmegaConf.merge(
                        cfg.agent, cfg.groups[group_name].props))
                    rewards = agent_cfg.get("rewards", {})
                    del agent_cfg["rewards"]
                    for inv_item in InventoryItemNames:
                        rewards[inv_item] = rewards.get(inv_item, 0)
                        rewards[inv_item + "_max"] = rewards.get(inv_item + "_max", 1000)
                    group_id = cfg.groups[group_name].id
                    agent = new Agent(
                        r, c, group_name, group_id, agent_cfg, rewards)
                    self._grid.add_object(agent)
                    agent.agent_id = self._agents.size()
                    self.add_agent(agent)
                    self._group_sizes[group_id] += 1

                if converter != NULL:
                    stat = "objects." + map[r,c]
                    self._stats.incr(stat)
                    self._grid.add_object(converter)
                    converter.set_event_manager(&self._event_manager)
                    converter = NULL



    cpdef list[str] grid_features(self):
        return self._grid_features

    def render(self):
        grid = self.render_ascii(["A", "#", "g", "c", "a"])
        for r in grid:
                print("".join(r))

    cpdef grid_objects(self):
        cdef GridObject *obj
        cdef ObsType[:] obj_data = np.zeros(len(self.grid_features()), dtype=self._obs_encoder.obs_np_type())
        cdef unsigned int obj_id, i
        cdef ObservationEncoder obs_encoder = self._obs_encoder
        cdef vector[unsigned int] offsets
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
            offsets.resize(obs_encoder._type_feature_names[obj._type_id].size())
            for i in range(offsets.size()):
                offsets[i] = i
            obs_encoder._encode(obj, obj_data, offsets)
            for i, name in enumerate(obs_encoder._type_feature_names[obj._type_id]):
                objects[obj_id][name] = obj_data[i]

        for agent_idx in range(self._agents.size()):
            agent_object = objects[self._agents[agent_idx].id]
            agent_object["agent_id"] = agent_idx

        return objects

    cpdef tuple[cnp.ndarray, cnp.ndarray, cnp.ndarray, cnp.ndarray, dict] step(self, cnp.ndarray actions):
        (obs, rewards, terms, truncs, infos) = super(MettaGrid, self).step(actions)

        self._group_rewards[:] = 0
        cdef Agent *agent
        cdef unsigned int group_id
        cdef float group_reward
        cdef bint share_rewards = False

        for agent_idx in range(self._agents.size()):
            if rewards[agent_idx] != 0:
                share_rewards = True
                agent = <Agent*>self._agents[agent_idx]
                group_id = agent.group
                group_reward = rewards[agent_idx] * self._group_reward_pct[group_id]
                rewards[agent_idx] -= group_reward
                self._group_rewards[group_id] += group_reward / self._group_sizes[group_id]

        if share_rewards:
            for agent_idx in range(self._agents.size()):
                agent = <Agent*>self._agents[agent_idx]
                group_id = agent.group
                group_reward = self._group_rewards[group_id]
                rewards[agent_idx] += group_reward
        return (obs, rewards, terms, truncs, infos)

    cpdef dict get_episode_stats(self):
        return {
            "game": self._stats.stats(),
            "agent": [ (<Agent*>agent).stats.stats() for agent in self._agents ]
        }
