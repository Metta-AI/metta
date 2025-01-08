
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

cdef class MettaGrid(GridEnv):
    cdef:
        object _cfg
        int _num_teams
        list _team_to_agents

    def __init__(self, cfg: OmegaConf, map: np.ndarray):
        cfg = OmegaConf.create(cfg)
        self._cfg = cfg

        obs_encoder = MettaObservationEncoder()
        if cfg.compact_obs:
            obs_encoder = MettaCompactObservationEncoder()

        actions = []
        if cfg.game.actions.noop.enabled:
            actions.append(Noop(cfg.game.actions.noop))
        if cfg.game.actions.move.enabled:
            actions.append(Move(cfg.game.actions.move))
        if cfg.game.actions.rotate.enabled:
            actions.append(Rotate(cfg.game.actions.rotate))
        if cfg.game.actions.use.enabled:
            actions.append(Use(cfg.game.actions.use))
        if cfg.game.actions.attack.enabled:
            actions.append(Attack(cfg.game.actions.attack))
        if cfg.game.actions.shield.enabled:
            actions.append(Shield(cfg.game.actions.shield))
        if cfg.game.actions.gift.enabled:
            actions.append(Gift(cfg.game.actions.gift))
        if cfg.game.actions.swap.enabled:
            actions.append(Swap(cfg.game.actions.swap))

        GridEnv.__init__(
            self,
            cfg.game.num_agents,
            map.shape[1],
            map.shape[0],
            cfg.game.max_steps,
            dict(ObjectLayers).values(),
            cfg.game.obs_width, cfg.game.obs_height,
            obs_encoder,
            actions,
            [ ResetHandler() ],
            use_flat_actions=cfg.flatten_actions,
            track_last_action=cfg.track_last_action
        )

        cdef Agent *agent
        for r in range(map.shape[0]):
            for c in range(map.shape[1]):
                if map[r,c] == "W":
                    self._grid.add_object(new Wall(r, c, cfg.game.objects.wall))
                    self._stats.game_incr("objects.wall")
                elif map[r,c] == "g":
                    self._grid.add_object(new Generator(r, c, cfg.game.objects.generator))
                    self._stats.game_incr("objects.generator")
                elif map[r,c] == "c":
                    self._grid.add_object(new Converter(r, c, cfg.game.objects.converter))
                    self._stats.game_incr("objects.converter")
                elif map[r,c] == "a":
                    self._grid.add_object(new Altar(r, c, cfg.game.objects.altar))
                    self._stats.game_incr("objects.altar")
                elif map[r,c][0] == "A":
                    agent = new Agent(r, c, cfg.game.objects.agent)
                    self._grid.add_object(agent)
                    self.add_agent(agent)
                    self._stats.game_incr("objects.agent")

        # Assign team to agents for kinship rewards sharing.
        if cfg.kinship.enabled:
            team = 0
            in_team = 0
            # Shuffle agent indices to randomize team assignment.
            indices = np.arange(0, self._agents.size())
            np.random.shuffle(indices)
            for id in indices:
                (<Agent*>self._agents[id]).team = team
                in_team += 1
                if in_team == cfg.kinship.team_size:
                    in_team = 0
                    team += 1
            self._num_teams = team + 1
            self._team_to_agents = [[] for i in range(self._num_teams)]
            for id in range(self._agents.size()):
                team = (<Agent*>self._agents[id]).team
                self._team_to_agents[team].append(id)

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
            agent_object["team"] = (<Agent*>self._agents[agent_idx]).team

        return objects

    cpdef tuple[cnp.ndarray, cnp.ndarray, cnp.ndarray, cnp.ndarray, dict] step(self, cnp.ndarray actions):
        (obs, rewards, terms, truncs, infos) = super(MettaGrid, self).step(actions)

        if self._cfg.kinship.enabled and self._cfg.kinship.team_reward > 0:
            team_rewards = np.zeros(self._num_teams + 1)
            for agent_idx in range(self._agents.size()):
                team = (<Agent*>self._agents[agent_idx]).team
                team_rewards[team] += self._cfg.kinship.team_reward * rewards[agent_idx]
                rewards[agent_idx] -= self._cfg.kinship.team_reward * rewards[agent_idx]

            team_idxs = team_rewards.nonzero()[0]
            for team in team_idxs:
                team_agents = self._team_to_agents[team]
                team_reward = team_rewards[team] / len(team_agents)
                rewards[team_agents] += team_reward

        return (obs, rewards, terms, truncs, infos)
