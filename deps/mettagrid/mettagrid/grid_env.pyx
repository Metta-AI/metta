from libc.stdio cimport printf
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp
import gymnasium as gym

from mettagrid.action_handler cimport ActionArg, ActionHandler
from mettagrid.grid cimport Grid
from mettagrid.grid_object cimport (
    GridObject,
    GridObjectId,
    Layer,
    GridLocation
)
from mettagrid.objects.agent cimport Agent
from mettagrid.observation_encoder cimport ObsType
from mettagrid.objects.production_handler cimport ProductionHandler, CoolDownHandler
from mettagrid.objects.constants cimport ObjectTypeNames, ObjectTypeAscii, InventoryItemNames

# Constants
obs_np_type = np.uint8

cdef class GridEnv:
    def __init__(
            self,
            unsigned int max_agents,
            unsigned int map_width,
            unsigned int map_height,
            unsigned int max_timestep,
            vector[Layer] layer_for_type_id,
            unsigned short obs_width,
            unsigned short obs_height
        ):
        self._obs_width = obs_width
        self._obs_height = obs_height
        self._max_timestep = max_timestep
        self._current_timestep = 0
        self._grid = new Grid(map_width, map_height, layer_for_type_id)
        self._grid_features = self._obs_encoder.feature_names()

        self._event_manager.init(self._grid, &self._stats)
        # The order of this needs to match the order in the Events enum
        self._event_manager.event_handlers.push_back(new ProductionHandler(&self._event_manager))
        self._event_manager.event_handlers.push_back(new CoolDownHandler(&self._event_manager))

        self.set_buffers(
            np.zeros(
                (
                    max_agents,
                    self._grid_features.size(),
                    self._obs_height,
                    self._obs_width
                ),
                dtype=obs_np_type),
            np.zeros(max_agents, dtype=np.int8),
            np.zeros(max_agents, dtype=np.int8),
            np.zeros(max_agents, dtype=np.float32)
        )


        self._action_success = vector[bint](max_agents)

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
        return (self._observations_np, self._rewards_np, self._terminals_np, self._truncations_np, {})

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
