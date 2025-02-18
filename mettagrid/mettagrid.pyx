from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map

import numpy as np
cimport numpy as cnp
import gymnasium as gym
from omegaconf import OmegaConf
from types import SimpleNamespace

from mettagrid.grid_env cimport GridEnv
from mettagrid.grid_object cimport GridObject
from mettagrid.observation_encoder cimport ObsType

from mettagrid.objects cimport ObjectLayers, Mine, Agent, ResetHandler, Wall, Generator, Altar, Lab, Factory, Temple, Armory, Lasery, Usable
from mettagrid.observation_encoder cimport MettaObservationEncoder, MettaCompactObservationEncoder
from mettagrid.actions.move import Move
from mettagrid.actions.rotate import Rotate
from mettagrid.actions.use import Use
from mettagrid.actions.attack import Attack
from mettagrid.actions.attack_nearest import AttackNearest
from mettagrid.actions.noop import Noop
from mettagrid.actions.swap import Swap
from mettagrid.actions.change_color import ChangeColorAction
from mettagrid.objects cimport InventoryItemNames
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
            [ ResetHandler() ],
            use_flat_actions=env_cfg.flatten_actions,
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
        cdef string group_name
        cdef unsigned char group_id
        for r in range(map.shape[0]):
            for c in range(map.shape[1]):

                if map[r,c] == "wall":
                    self._grid.add_object(new Wall(r, c, cfg.objects.wall))
                    self._stats.incr(b"objects.wall")

                elif map[r,c] == "mine":
                    self._grid.add_object(new Mine(r, c, cfg.objects.mine))
                    self._stats.incr(b"objects.mine")
                elif map[r,c] == "generator":
                    self._grid.add_object(new Generator(r, c, cfg.objects.generator))
                    self._stats.incr(b"objects.generator")
                elif map[r,c] == "altar":
                    self._grid.add_object(new Altar(r, c, cfg.objects.altar))
                    self._stats.incr(b"objects.altar")
                elif map[r,c] == "armory":
                    self._grid.add_object(new Armory(r, c, cfg.objects.armory))
                    self._stats.incr(b"objects.armory")
                elif map[r,c] == "lasery":
                    self._grid.add_object(new Lasery(r, c, cfg.objects.lasery))
                    self._stats.incr(b"objects.lasery")

                elif map[r,c] == "lab":
                    self._grid.add_object(new Lab(r, c, cfg.objects.lab))
                    self._stats.incr(b"objects.lab")

                elif map[r,c] == "factory":
                    self._grid.add_object(new Factory(r, c, cfg.objects.factory))
                    self._stats.incr(b"objects.factory")

                elif map[r,c] == "temple":
                    self._grid.add_object(new Temple(r, c, cfg.objects.temple))
                    self._stats.incr(b"objects.temple")

                elif map[r,c].startswith("agent."):
                    group_name = map[r,c].split(".")[1]
                    agent_cfg = OmegaConf.to_container(OmegaConf.merge(
                        cfg.agent, cfg.groups[group_name].props))
                    rewards = agent_cfg.get("rewards", {})
                    del agent_cfg["rewards"]
                    for inv_item in InventoryItemNames:
                        rewards[inv_item] = rewards.get(inv_item, 0)
                    group_id = cfg.groups[group_name].id
                    agent = new Agent(
                        r, c, group_name, group_id, agent_cfg, rewards)
                    self._grid.add_object(agent)
                    agent.agent_id = self._agents.size()
                    self.add_agent(agent)
                    self._group_sizes[group_id] += 1
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
            print(actions[agent_idx])

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
