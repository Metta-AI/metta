from libc.stdio cimport printf
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
import numpy as np
cimport numpy as cnp
import gymnasium as gym
from omegaconf import DictConfig, ListConfig, OmegaConf

from mettagrid.event cimport EventManager
from mettagrid.stats_tracker cimport StatsTracker

from mettagrid.action_handler cimport ActionArg, ActionHandler
from mettagrid.grid cimport Grid
from mettagrid.grid_object cimport (
    GridObject,
    GridObjectId,
    GridLocation
)
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.wall cimport Wall
from mettagrid.objects.converter cimport Converter
from mettagrid.observation_encoder cimport (
    ObsType,
    ObservationEncoder
)
from mettagrid.grid_object cimport GridObject
from mettagrid.objects.production_handler cimport ProductionHandler, CoolDownHandler
from mettagrid.objects.constants cimport ObjectLayers, ObjectTypeNames, ObjectTypeAscii, InventoryItemNames, ObjectType

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

# Constants
obs_np_type = np.uint8
ctypedef unsigned int ActionType

cdef class MettaGrid:
    cdef:
        object _cfg
        map[unsigned int, float] _group_reward_pct
        map[unsigned int, unsigned int] _group_sizes
        cnp.ndarray _group_rewards_np
        double[:] _group_rewards
        Grid *_grid
        EventManager _event_manager
        unsigned int _current_timestep
        unsigned int _max_timestep

        vector[ActionHandler*] _action_handlers
        int _num_action_handlers
        vector[unsigned char] _max_action_args
        unsigned char _max_action_arg
        unsigned char _max_action_priority

        ObservationEncoder _obs_encoder
        StatsTracker _stats

        unsigned short _obs_width
        unsigned short _obs_height

        vector[Agent*] _agents

        cnp.ndarray _observations_np
        ObsType[:,:,:,:] _observations
        cnp.ndarray _terminals_np
        char[:] _terminals
        cnp.ndarray _truncations_np
        char[:] _truncations
        cnp.ndarray _rewards_np
        float[:] _rewards
        cnp.ndarray _episode_rewards_np
        float[:] _episode_rewards

        vector[string] _grid_features

        bint _track_last_action
        unsigned char _last_action_obs_idx
        unsigned char _last_action_arg_obs_idx
        vector[bint] _action_success

    def __init__(self, env_cfg: DictConfig | ListConfig, map: np.ndarray):
        cfg = OmegaConf.create(env_cfg.game)
        self._cfg = cfg
        num_agents = cfg.num_agents
        max_timestep = cfg.max_steps
        layer_for_type_id = dict(ObjectLayers).values()
        obs_width = cfg.obs_width
        obs_height = cfg.obs_height

        self._obs_width = obs_width
        self._obs_height = obs_height
        self._max_timestep = max_timestep
        self._current_timestep = 0
        self._grid = new Grid(map.shape[1], map.shape[0], layer_for_type_id)
        self._grid_features = self._obs_encoder.feature_names()

        self._event_manager.init(self._grid, &self._stats)
        # The order of this needs to match the order in the Events enum
        self._event_manager.event_handlers.push_back(new ProductionHandler(&self._event_manager))
        self._event_manager.event_handlers.push_back(new CoolDownHandler(&self._event_manager))

        self.set_buffers(
            np.zeros(
                (
                    num_agents,
                    self._grid_features.size(),
                    self._obs_height,
                    self._obs_width
                ),
                dtype=obs_np_type),
            np.zeros(num_agents, dtype=np.int8),
            np.zeros(num_agents, dtype=np.int8),
            np.zeros(num_agents, dtype=np.float32)
        )

        self._action_success.resize(num_agents)

        # Set up action handlers. This would be cleaner in a separate function. We're leaving
        # it inline since we're moving to reduce cython, and don't want to have to add more
        # Python functions just now (which this would need to be, since we'd be passing the cfg).
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
        
        self.init_action_handlers(actions)

        self._group_rewards_np = np.zeros(len(cfg.groups))
        self._group_rewards = self._group_rewards_np
        self._group_sizes = {
            g.id: 0 for g in cfg.groups.values()
        }
        self._group_reward_pct = {
            g.id: g.get("group_reward_pct", 0) for g in cfg.groups.values()
        }

        # Initialize objects from the map.
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

    def __dealloc__(self):
        del self._grid

    cdef void init_action_handlers(self, vector[ActionHandler*] action_handlers):
        """Initializes action_handlers.

        This lives separate from __init__ since
          * __init__ is a Python function, and so only Python objects can be passed
          * ActionHandlers are cpp objects, not Python objects
        """
        self._action_handlers = action_handlers
        self._num_action_handlers = action_handlers.size()
        self._max_action_priority = 0
        self._max_action_arg = 0
        self._max_action_args.resize(action_handlers.size())
        cdef ActionHandler *handler
        cdef unsigned int i
        for i in range(action_handlers.size()):
            handler = action_handlers[i]
            handler.init(self._grid)
            max_arg = handler.max_arg()
            self._max_action_args[i] = max_arg
            self._max_action_arg = max(self._max_action_arg, max_arg)
            self._max_action_priority = max(self._max_action_priority, handler.priority)


    cdef void add_agent(self, Agent* agent):
        agent.init(&self._rewards[self._agents.size()])
        self._agents.push_back(agent)

    cdef void _compute_observation(
        self,
        unsigned observer_r, unsigned int observer_c,
        unsigned short obs_width, unsigned short obs_height,
        ObsType[:,:,:] observation):

        cdef:
            int r, c, layer
            GridLocation object_loc
            GridObject *obj
            unsigned short obs_width_r = obs_width >> 1
            unsigned short obs_height_r = obs_height >> 1
            cdef unsigned int obs_r, obs_c
            cdef ObsType[:] agent_ob

        cdef unsigned int r_start = max(observer_r, obs_height_r) - obs_height_r
        cdef unsigned int c_start = max(observer_c, obs_width_r) - obs_width_r
        for r in range(r_start, observer_r + obs_height_r + 1):
            if r < 0 or r >= self._grid.height:
                continue
            for c in range(c_start, observer_c + obs_width_r + 1):
                if c < 0 or c >= self._grid.width:
                    continue
                for layer in range(self._grid.num_layers):
                    object_loc = GridLocation(r, c, layer)
                    obj = self._grid.object_at(object_loc)
                    if obj == NULL:
                        continue

                    obs_r = object_loc.r + obs_height_r - observer_r
                    obs_c = object_loc.c + obs_width_r - observer_c
                    agent_ob = observation[obs_r, obs_c, :]
                    self._obs_encoder.encode(obj, &agent_ob[0])

    cdef void _compute_observations(self, int[:,:] actions):
        cdef Agent *agent
        for idx in range(self._agents.size()):
            agent = self._agents[idx]
            self._compute_observation(
                agent.location.r,
                agent.location.c,
                self._obs_width,
                self._obs_height,
                self._observations[idx]
            )

    cdef void _step(self, int[:,:] actions):
        cdef:
            unsigned int idx
            short action
            ActionArg arg
            Agent *agent
            ActionHandler *handler

        self._rewards[:] = 0
        self._observations[:, :, :, :] = 0

        # Clear the success flags.
        for i in range(self._action_success.size()):
            self._action_success[i] = 0

        self._current_timestep += 1
        self._event_manager.process_events(self._current_timestep)

        cdef unsigned char p
        for p in range(self._max_action_priority + 1):
            for idx in range(self._agents.size()):
                action = actions[idx][0]
                if action < 0 or action >= self._num_action_handlers:
                    printf("Invalid action: %d\n", action)
                    continue

                arg = actions[idx][1]
                agent = self._agents[idx]
                handler = self._action_handlers[action]
                if handler.priority != self._max_action_priority - p:
                    continue
                if arg > self._max_action_args[action]:
                    continue
                self._action_success[idx] = handler.handle_action(idx, agent.id, arg, self._current_timestep)

        self._compute_observations(actions)

        for i in range(self._episode_rewards.shape[0]):
            self._episode_rewards[i] += self._rewards[i]

        if self._max_timestep > 0 and self._current_timestep >= self._max_timestep:
            self._truncations[:] = 1

    ###############################
    # Python API
    ###############################
    cpdef tuple[cnp.ndarray, dict] reset(self):
        if self._current_timestep > 0:
            raise NotImplemented("Cannot reset after stepping")

        self._terminals[:] = 0
        self._truncations[:] = 0
        self._episode_rewards[:] = 0
        self._observations[:, :, :, :] = 0
        self._rewards[:] = 0

        self._compute_observations(np.zeros((self._agents.size(), 2), dtype=np.int32))
        return (self._observations_np, {})

    cpdef tuple[cnp.ndarray, cnp.ndarray, cnp.ndarray, cnp.ndarray, dict] step(self, cnp.ndarray actions):
        self._step(actions)
        (obs, rewards, terms, truncs, infos) = (self._observations_np, self._rewards_np, self._terminals_np, self._truncations_np, {})

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

    cpdef void set_buffers(
        self,
        cnp.ndarray[ObsType, ndim=4] observations,
        cnp.ndarray[char, ndim=1] terminals,
        cnp.ndarray[char, ndim=1] truncations,
        cnp.ndarray[float, ndim=1] rewards):

        self._observations_np = observations
        self._observations = observations
        self._terminals_np = terminals
        self._terminals = terminals
        self._truncations_np = truncations
        self._truncations = truncations
        self._rewards_np = rewards
        self._rewards = rewards
        self._episode_rewards_np = np.zeros_like(rewards)
        self._episode_rewards = self._episode_rewards_np

        for i in range(self._agents.size()):
            self._agents[i].init(&self._rewards[i])
    
    cpdef grid(self):
        return []
    
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

    cpdef list[str] action_names(self):
        return [handler.action_name() for handler in self._action_handlers]

    cpdef unsigned int current_timestep(self):
        return self._current_timestep

    cpdef unsigned int map_width(self):
        return self._grid.width

    cpdef unsigned int map_height(self):
        return self._grid.height

    cpdef list[str] grid_features(self):
        return self._grid_features

    cpdef unsigned int num_agents(self):
        return self._agents.size()

    cpdef observe(
        self,
        GridObjectId observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        ObsType[:,:,:] observation):

        cdef GridObject* observer = self._grid.object(observer_id)
        self._compute_observation(
            observer.location.r, observer.location.c, obs_width, obs_height, observation)

    cpdef observe_at(
        self,
        unsigned short row,
        unsigned short col,
        unsigned short obs_width,
        unsigned short obs_height,
        ObsType[:,:,:] observation):

        self._compute_observation(
            row, col, obs_width, obs_height, observation)

    cpdef get_episode_rewards(self):
        return self._episode_rewards_np

    cpdef dict get_episode_stats(self):
        return {
            "game": self._stats.stats(),
            "agent": [ (<Agent*>agent).stats.stats() for agent in self._agents ]
        }

    cpdef cnp.ndarray render_ascii(self):
        cdef GridObject *obj
        grid = np.full((self._grid.height, self._grid.width), " ", dtype=np.str_)
        for obj_id in range(1, self._grid.objects.size()):
            obj = self._grid.object(obj_id)
            grid[obj.location.r, obj.location.c] = ObjectTypeAscii[obj._type_id]
        return grid

    @property
    def action_space(self):
        return gym.spaces.MultiDiscrete((len(self.action_names()), self._max_action_arg + 1), dtype=np.int64)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            0,
            255,
            shape=(self._obs_height, self._obs_width, self._grid_features.size()),
            dtype=obs_np_type
        )

    def action_success(self):
        return self._action_success

    def max_action_args(self):
        return self._max_action_args

    def object_type_names(self):
        return ObjectTypeNames

    def inventory_item_names(self):
        return InventoryItemNames
    
    def render(self):
        grid = self.render_ascii()
        for r in grid:
                print("".join(r))
