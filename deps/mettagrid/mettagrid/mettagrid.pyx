import numpy as np
cimport numpy as cnp
from omegaconf import DictConfig, ListConfig, OmegaConf

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
    ObservationEncoder
)

# Object imports
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.wall cimport Wall
from mettagrid.objects.converter cimport Converter
from mettagrid.objects.constants cimport ObjectLayers, InventoryItemNames, ObjectType

# Action imports
from mettagrid.action_handler cimport ActionHandler
from mettagrid.actions.move cimport Move
from mettagrid.actions.rotate cimport Rotate
from mettagrid.actions.get_output cimport GetOutput
from mettagrid.actions.put_recipe_items cimport PutRecipeItems
from mettagrid.actions.attack cimport Attack
from mettagrid.actions.attack_nearest cimport AttackNearest
from mettagrid.actions.noop cimport Noop
from mettagrid.actions.swap cimport Swap
from mettagrid.actions.change_color cimport ChangeColorAction

cdef class MettaGrid(GridEnv):
    cdef:
        object _cfg
        map[unsigned int, float] _group_reward_pct
        map[unsigned int, unsigned int] _group_sizes
        cnp.ndarray _group_rewards_np
        double[:] _group_rewards

    def __init__(self, env_cfg: DictConfig | ListConfig, map: np.ndarray):
        cfg = OmegaConf.create(env_cfg.game)
        self._cfg = cfg

        cdef vector[ActionHandler*] actions
        if cfg.actions.put_items.enabled:
            actions.push_back(new PutRecipeItems(cfg.actions.put_items))
        if cfg.actions.get_items.enabled:
            actions.push_back(new GetOutput(cfg.actions.get_items))
        if cfg.actions.noop.enabled:
            actions.push_back(new Noop(cfg.actions.noop))
        if cfg.actions.move.enabled:
            actions.push_back(new Move(cfg.actions.move))
        if cfg.actions.rotate.enabled:
            actions.push_back(new Rotate(cfg.actions.rotate))
        if cfg.actions.attack.enabled:
            actions.push_back(new Attack(cfg.actions.attack))
            actions.push_back(new AttackNearest(cfg.actions.attack))
        if cfg.actions.swap.enabled:
            actions.push_back(new Swap(cfg.actions.swap))
        if cfg.actions.change_color.enabled:
            actions.push_back(new ChangeColorAction(cfg.actions.change_color))

        GridEnv.__init__(
            self,
            cfg.num_agents,
            map.shape[1],
            map.shape[0],
            cfg.max_steps,
            dict(ObjectLayers).values(),
            cfg.obs_width, cfg.obs_height
        )
        self.init_action_handlers(actions)

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
                elif map[r,c] == "block":
                    block = new Wall(r, c, cfg.objects.block)
                    self._grid.add_object(block)
                    self._stats.incr(b"objects.block")
                elif map[r,c].startswith("mine"):
                    m = map[r,c]
                    if "." not in m:
                        m = "mine.red"
                    converter = new Converter(r, c, cfg.objects[m], ObjectType.MineT)
                elif map[r,c].startswith("generator"):
                    m = map[r,c]
                    if "." not in m:
                        m = "generator.red"
                    converter = new Converter(r, c, cfg.objects[m], ObjectType.GeneratorT)
                elif map[r,c] == "altar":
                    converter = new Converter(r, c, cfg.objects.altar, ObjectType.AltarT)
                elif map[r,c] == "armory":
                    converter = new Converter(r, c, cfg.objects.armory, ObjectType.ArmoryT)
                elif map[r,c] == "lasery":
                    converter = new Converter(r, c, cfg.objects.lasery, ObjectType.LaseryT)
                elif map[r,c] == "lab":
                    converter = new Converter(r, c, cfg.objects.lab, ObjectType.LabT)
                elif map[r,c] == "factory":
                    converter = new Converter(r, c, cfg.objects.factory, ObjectType.FactoryT)
                elif map[r,c] == "temple":
                    converter = new Converter(r, c, cfg.objects.temple, ObjectType.TempleT)
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


    def render(self):
        grid = self.render_ascii()
        for r in grid:
                print("".join(r))

    cpdef grid_objects(self):
        cdef GridObject *obj
        cdef ObsType[:] obj_data = np.zeros(len(self.grid_features()), dtype=np.uint8)
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
            # We want observations written to our vector, rather than "normal" observation
            # space, so we need to build our own offsets.
            offsets.resize(obs_encoder.type_feature_names()[obj._type_id].size())
            for i in range(offsets.size()):
                offsets[i] = i
            obs_encoder.encode(obj, &obj_data[0], offsets)
            for i, name in enumerate(obs_encoder.type_feature_names()[obj._type_id]):
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
