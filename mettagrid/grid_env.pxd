cimport numpy as cnp
import numpy as np

from libcpp.string cimport string
from libcpp.vector cimport vector
from mettagrid.action cimport ActionHandler, Action
from mettagrid.event cimport EventManager
from mettagrid.stats_tracker cimport StatsTracker
from mettagrid.grid_object cimport GridObjectId, GridObject
from mettagrid.grid cimport Grid
from mettagrid.event cimport EventManager
from mettagrid.base_encoder cimport ObservationEncoder, ObsType

from libc.stdio cimport printf

ctypedef unsigned int ActionType

cdef class GridEnv:
    cdef:
        Grid *_grid
        EventManager _event_manager
        unsigned int _current_timestep
        unsigned int _max_timestep

        list[ActionHandler] _action_handlers
        int _num_action_handlers
        vector[unsigned char] _max_action_args
        unsigned char _max_action_arg
        unsigned char _max_action_priority

        bint _use_flat_actions
        vector[Action] _flat_actions

        ObservationEncoder _obs_encoder

        unsigned short _obs_width
        unsigned short _obs_height
        unsigned short _middle_x
        unsigned short _middle_y

        vector[GridObject*] _agents

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

        StatsTracker _stats

        vector[string] _grid_features

        bint _track_last_action
        unsigned char _last_action_obs_idx
        unsigned char _last_action_arg_obs_idx

    cdef void add_agent(self, GridObject* agent)

    cdef void _compute_observations(self, int[:,:] actions)
    cdef void _step(self, int[:,:] actions)

    cdef void _compute_observation(
        self,
        unsigned int observer_r,
        unsigned int observer_c,
        unsigned short obs_width,
        unsigned short obs_height,
        ObsType[:,:,:] observation)

    cdef cnp.ndarray _unflatten_actions(self, cnp.ndarray actions)

    ############################################
    # Python API
    ############################################

    cpdef void set_buffers(
        self,
        cnp.ndarray[ObsType, ndim=4] observations,
        cnp.ndarray[char, ndim=1] terminals,
        cnp.ndarray[char, ndim=1] truncations,
        cnp.ndarray[float, ndim=1] rewards)


    cpdef grid(self)
    cpdef unsigned int current_timestep(self)
    cpdef unsigned int map_width(self)
    cpdef unsigned int map_height(self)
    cpdef list[str] grid_features(self)
    cpdef list[str] action_names(self)
    cpdef unsigned int num_agents(self)
    cpdef tuple observation_shape(self)

    cpdef tuple[cnp.ndarray, dict] reset(self)
    cpdef tuple[cnp.ndarray, cnp.ndarray, cnp.ndarray, cnp.ndarray, dict] step(self, cnp.ndarray actions)

    cpdef observe(
        self,
        GridObjectId observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        ObsType[:,:,:] observation)

    cpdef observe_at(
        self,
        unsigned short row,
        unsigned short col,
        unsigned short obs_width,
        unsigned short obs_height,
        ObsType[:,:,:] observation)

    cpdef get_episode_stats(self)
    cpdef get_episode_rewards(self)

    cpdef tuple get_buffers(self)
    cpdef cnp.ndarray render_ascii(self, list[char] type_to_char)
    cpdef cnp.ndarray grid_objects_types(self)
    cpdef cnp.ndarray unflatten_actions(self, cnp.ndarray actions)
    cpdef cnp.ndarray flatten_actions(self, cnp.ndarray actions)
