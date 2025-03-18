from libc.stdio cimport printf
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp
import gymnasium as gym

from mettagrid.action cimport ActionArg, ActionHandler
from mettagrid.event cimport EventManager, EventHandler
from mettagrid.grid cimport Grid
from mettagrid.grid_object cimport (
    GridObject,
    GridObjectId,
    Layer,
    GridLocation
)
from mettagrid.observation_encoder cimport ObservationEncoder, ObsType
from mettagrid.objects.production_handler cimport ProductionHandler, CoolDownHandler

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
            unsigned short obs_height,
            ObservationEncoder observation_encoder,
            list[ActionHandler] action_handlers,
            bint track_last_action=False
        ):
        self._obs_width = obs_width
        self._obs_height = obs_height
        self._middle_x = obs_width // 2
        self._middle_y = obs_height // 2
        self._max_timestep = max_timestep
        self._current_timestep = 0
        self._grid = new Grid(map_width, map_height, layer_for_type_id)
        self._obs_encoder = observation_encoder
        self._obs_encoder.init(self._obs_width, self._obs_height)
        self._grid_features = observation_encoder.feature_names()

        if self._track_last_action:
            self._last_action_obs_idx = self._grid_features.size()
            self._grid_features.push_back(b"last_action")
            self._last_action_arg_obs_idx = self._grid_features.size()
            self._grid_features.push_back(b"last_action_argument")

        self._action_handlers = action_handlers
        self._num_action_handlers = len(action_handlers)
        self._max_action_priority = 0
        self._max_action_arg = 0
        self._max_action_args.resize(len(action_handlers))
        for i, handler in enumerate(action_handlers):
            (<ActionHandler>handler).init(self)
            max_arg = (<ActionHandler>handler).max_arg()
            self._max_action_args[i] = max_arg
            self._max_action_arg = max(self._max_action_arg, max_arg)
            self._max_action_priority = max(self._max_action_priority, (<ActionHandler>handler)._priority)

        self._event_manager.init(self._grid, &self._stats)
        # The order of this needs to match the order in the Events enum
        self._event_manager.event_handlers.push_back(new ProductionHandler(&self._event_manager))
        self._event_manager.event_handlers.push_back(new CoolDownHandler(&self._event_manager))
        self._track_last_action = track_last_action

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

    cdef void add_agent(self, GridObject* agent):
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
                    self._obs_encoder.encode(obj, agent_ob)

    cdef void _compute_observations(self, int[:,:] actions):
        cdef GridObject *agent
        for idx in range(self._agents.size()):
            agent = self._agents[idx]
            self._compute_observation(
                agent.location.r,
                agent.location.c,
                self._obs_width,
                self._obs_height,
                self._observations[idx]
            )

        if self._track_last_action:
            for idx in range(self._agents.size()):
                self._observations[idx][22][self._middle_y][self._middle_x] = actions[idx][0]
                self._observations[idx][23][self._middle_y][self._middle_x] = actions[idx][1]

    cdef void _step(self, int[:,:] actions):
        cdef:
            unsigned int idx
            short action
            ActionArg arg
            GridObject *agent
            ActionHandler handler

        self._rewards[:] = 0
        self._observations[:, :, :, :] = 0

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
                handler = <ActionHandler>self._action_handlers[action]
                if handler._priority != self._max_action_priority - p:
                    continue
                if arg > self._max_action_args[action]:
                    continue
                self._action_success[idx] = handler.handle_action(idx, agent.id, arg)

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

    cpdef tuple observation_shape(self):
        return (len(self.grid_features()), self.obs_height, self.obs_width)

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

    cpdef tuple get_buffers(self):
        return (self._observations_np, self._terminals_np, self._truncations_np, self._rewards_np)

    cpdef cnp.ndarray render_ascii(self, list[char] type_to_char):
        cdef GridObject *obj
        grid = np.full((self._grid.height, self._grid.width), " ", dtype=np.str_)
        for obj_id in range(1, self._grid.objects.size()):
            obj = self._grid.object(obj_id)
            grid[obj.location.r, obj.location.c] = type_to_char[obj._type_id]
        return grid

    cpdef cnp.ndarray grid_objects_types(self):
        cdef GridObject *obj
        grid = np.zeros((self._grid.height, self._grid.width), dtype=np.uint8)
        for obj_id in range(1, self._grid.objects.size()):
            obj = self._grid.object(obj_id)
            grid[obj.location.r, obj.location.c] = obj._type_id + 1
        return grid

    @property
    def action_space(self):
        return gym.spaces.MultiDiscrete((len(self.action_names()), self._max_action_arg), dtype=np.int64)

    @property
    def observation_space(self):
        space = self._obs_encoder.observation_space()
        return gym.spaces.Box(
            0,
            255,
            shape=(self._obs_height, self._obs_width, self._grid_features.size()),
            dtype=obs_np_type
        )

    def action_success(self):
        return self._action_success
