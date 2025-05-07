# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int8_t, int32_t

# Python imports
import json
import numpy as np
cimport numpy as cnp
import gymnasium as gym
from omegaconf import DictConfig, ListConfig, OmegaConf

# Import from the cython definition file
from mettagrid.py_mettagrid cimport MettaGrid, GridObjectId, ObsType

# Constants
obs_np_type = np.uint8

# Wrapper class for the C++ implementation
cdef class PyMettaGrid:
    cdef:
        object _cfg
        # The C++ implementation of MettaGrid
        MettaGrid* _cpp_mettagrid
        
        # NumPy array views for Python access
        cnp.ndarray _observations_np
        cnp.ndarray _terminals_np
        cnp.ndarray _truncations_np
        cnp.ndarray _rewards_np
        cnp.ndarray _episode_rewards_np
        cnp.ndarray _group_rewards_np
        
        # Cache for frequently accessed data
        list _grid_features_list
        uint32_t _feature_size
        
        # Pre-allocated buffers for frequent operations
        int32_t** _c_actions
        cnp.ndarray _action_array_buffer
        cnp.ndarray _obs_buffer
        uint32_t _num_agents


    def __init__(self, env_cfg: DictConfig | ListConfig, np_map: cnp.ndarray):
        # Initialize configuration
        cfg = OmegaConf.create(env_cfg.game)
        self._cfg = cfg
        
        # Extract parameters
        num_agents = cfg.num_agents
        self._num_agents = num_agents
        max_timestep = cfg.max_steps
        obs_width = cfg.obs_width
        obs_height = cfg.obs_height
        map_width = np_map.shape[1]
        map_height = np_map.shape[0]
        
        # Create the C++ MettaGrid instance with ownership of the grid
        self._cpp_mettagrid = new MettaGrid(
            map_width, map_height, num_agents, max_timestep, obs_width, obs_height
        )
        
        if self._cpp_mettagrid == NULL:
            raise MemoryError("Failed to allocate MettaGrid")
        
        # Initialize objects from the map and config
        cfg_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))
        map_json = json.dumps(np_map.tolist())
        self._cpp_mettagrid.initialize_from_json(
            map_json.encode('utf8'), 
            cfg_json.encode('utf8')
        )
        
        # Cache grid features
        self._grid_features_list = self._get_grid_features()
        self._feature_size = len(self._grid_features_list)
        
        # Set up NumPy array views for Python access
        self._create_numpy_views(num_agents, self._feature_size, obs_height, obs_width)
        
        # Pre-allocate action arrays for step and reset
        self._c_actions = <int32_t**>malloc(num_agents * sizeof(int32_t*))
        if self._c_actions == NULL:
            raise MemoryError("Failed to allocate action buffer")
            
        cdef uint32_t i
        for i in range(num_agents):
            self._c_actions[i] = <int32_t*>malloc(2 * sizeof(int32_t))
            if self._c_actions[i] == NULL:
                # Clean up already allocated memory
                for j in range(i):
                    free(self._c_actions[j])
                free(self._c_actions)
                raise MemoryError("Failed to allocate action buffer element")
        
        # Pre-allocate NumPy arrays for common operations
        self._action_array_buffer = np.zeros((num_agents, 2), dtype=np.int32)
        self._obs_buffer = np.zeros((obs_height, obs_width, self._feature_size), dtype=np.uint8)


    def __dealloc__(self):
        # Clean up pre-allocated action arrays
        if self._c_actions != NULL:
            for i in range(self._num_agents):
                if self._c_actions[i] != NULL:
                    free(self._c_actions[i])
            free(self._c_actions)
            self._c_actions = NULL
        
        # Clean up the C++ object
        if self._cpp_mettagrid != NULL:
            del self._cpp_mettagrid
            self._cpp_mettagrid = NULL


    def _create_numpy_views(self, uint32_t num_agents, uint32_t feature_size, uint16_t obs_height, uint16_t obs_width):
        """Create NumPy array views that reference the C++ internal buffers."""
        cdef:
            # For observations
            vector[ObsType] cpp_observations = self._cpp_mettagrid.get_observations()
            ObsType* obs_ptr = cpp_observations.data()
            size_t obs_size = cpp_observations.size()
            ObsType[:] obs_view = <ObsType[:obs_size]>obs_ptr
            
            # For terminals
            vector[int8_t] cpp_terminals = self._cpp_mettagrid.get_terminals()
            int8_t* term_ptr = cpp_terminals.data()
            size_t term_size = cpp_terminals.size()
            int8_t[:] terminals_view = <int8_t[:term_size]>term_ptr
            
            # For truncations
            vector[int8_t] cpp_truncations = self._cpp_mettagrid.get_truncations()
            int8_t* trunc_ptr = cpp_truncations.data()
            size_t trunc_size = cpp_truncations.size()
            int8_t[:] truncations_view = <int8_t[:trunc_size]>trunc_ptr
            
            # For rewards
            vector[float] cpp_rewards = self._cpp_mettagrid.get_rewards()
            float* reward_ptr = cpp_rewards.data()
            size_t reward_size = cpp_rewards.size()
            float[:] rewards_view = <float[:reward_size]>reward_ptr
            
            # For episode rewards
            vector[float] cpp_episode_rewards = self._cpp_mettagrid.get_episode_rewards()
            float* ep_reward_ptr = cpp_episode_rewards.data()
            size_t ep_reward_size = cpp_episode_rewards.size()
            float[:] episode_rewards_view = <float[:ep_reward_size]>ep_reward_ptr
            
            # For group rewards
            vector[double] cpp_group_rewards = self._cpp_mettagrid.get_group_rewards()
            double* group_reward_ptr = cpp_group_rewards.data()
            size_t group_reward_size = cpp_group_rewards.size()
            double[:] group_rewards_view = <double[:group_reward_size]>group_reward_ptr

        # Create NumPy arrays from memory views
        self._observations_np = np.asarray(obs_view).reshape(num_agents, obs_height, obs_width, feature_size)
        self._terminals_np = np.asarray(terminals_view).astype(np.int8)
        self._truncations_np = np.asarray(truncations_view).astype(np.int8)
        self._rewards_np = np.asarray(rewards_view).astype(np.float32)
        self._episode_rewards_np = np.asarray(episode_rewards_view).astype(np.float32)
        self._group_rewards_np = np.asarray(group_rewards_view).astype(np.float64)

    # Helper method to get grid features as Python list - caching the result
    cdef list _get_grid_features(self):
        cdef vector[string] features = self._cpp_mettagrid.grid_features()
        cdef list result = []
        cdef uint32_t i
        for i in range(features.size()):
            result.append(features[i].decode('utf8'))
        return result


    def reset(self):
        """Reset the environment and return initial observation."""
        # Call the C++ reset method
        self._cpp_mettagrid.reset()
        
        # Initialize all actions to zero
        cdef uint32_t i
        for i in range(self._num_agents):
            self._c_actions[i][0] = 0
            self._c_actions[i][1] = 0
        
        # Compute observations using pre-allocated arrays
        self._cpp_mettagrid.compute_observations(self._c_actions)
        
        return (self._observations_np, {})
    
    
    def step(self, actions):
        """Take a step in the environment with the given actions."""
        cdef:
            cnp.ndarray[cnp.int32_t, ndim=2] actions_array
            uint32_t i, rows, cols
        
        # Fast path for already compatible numpy arrays
        if isinstance(actions, np.ndarray) and actions.shape == (self._num_agents, 2) and actions.dtype == np.int32:
            actions_array = actions
        else:
            # Ensure actions is a properly shaped numpy array using our buffer
            self._action_array_buffer.fill(0)  # Reset buffer
            
            if isinstance(actions, np.ndarray):
                if actions.ndim == 1:
                    # Reshape to 2D if it's a 1D array
                    actions_array = actions.reshape(-1, 1)
                else:
                    actions_array = actions
            else:
                actions_array = np.asarray(actions, dtype=np.int32)
            
            # Copy available values to our pre-allocated buffer
            rows = min(actions_array.shape[0], self._num_agents)
            cols = min(actions_array.shape[1], 2)
            self._action_array_buffer[:rows, :cols] = actions_array[:rows, :cols]
            actions_array = self._action_array_buffer
        
        # Copy NumPy array values to our C action array
        for i in range(self._num_agents):
            self._c_actions[i][0] = actions_array[i, 0]
            self._c_actions[i][1] = actions_array[i, 1]
        
        # Take a step in the C++ implementation
        self._cpp_mettagrid.step(self._c_actions)
        
        # Process group rewards
        self._cpp_mettagrid.compute_group_rewards(<float*>self._rewards_np.data)
        
        return (self._observations_np, self._rewards_np, self._terminals_np, self._truncations_np, {})
    
        
    def action_success(self):
        """Get the action success information."""
        cdef:
            vector[int8_t] success = self._cpp_mettagrid.action_success()
            int8_t* data_ptr = success.data()
            size_t size = success.size()
            int8_t[:] success_view = <int8_t[:size]>data_ptr
        # Efficiently create numpy array directly from buffer
        return np.asarray(success_view)

    def max_action_args(self):
        """Get the maximum action arguments."""
        cdef:
            vector[uint8_t] max_args = self._cpp_mettagrid.max_action_args()
            uint8_t* data_ptr = max_args.data()
            size_t size = max_args.size()
            uint8_t[:] max_args_view = <uint8_t[:size]>data_ptr
        
        # Efficiently create numpy array directly from buffer
        return np.asarray(max_args_view)


    def current_timestep(self):
        """Get the current timestep."""
        return self._cpp_mettagrid.current_timestep()
    
    
    def map_width(self):
        """Get the width of the map."""
        return self._cpp_mettagrid.map_width()
    
    
    def map_height(self):
        """Get the height of the map."""
        return self._cpp_mettagrid.map_height()
    
    
    def grid_features(self):
        """Get the list of grid features."""
        # Return cached list instead of recomputing
        return self._grid_features_list
    
    
    def num_agents(self):
        """Get the number of agents."""
        return self._num_agents


    def _observe_internal(self,
                         uint16_t obs_width,
                         uint16_t obs_height,
                         observation,
                         bint is_observer_id=False,
                         GridObjectId observer_id=0,
                         uint16_t row=0,
                         uint16_t col=0):
        """
        Internal helper method for observation functions.
        
        Args:
            obs_width: Width of the observation window
            obs_height: Height of the observation window
            observation: Buffer to store the observation (numpy array)
            is_observer_id: Whether to observe from an object's perspective (True) or location (False)
            observer_id: ID of the observer object (used if is_observer_id is True)
            row: Row coordinate (used if is_observer_id is False)
            col: Column coordinate (used if is_observer_id is False)
        """
        # Setup the observation array
        cdef:
            cnp.ndarray obs_array
        
        if observation is None:
            # Use pre-allocated buffer when None is provided
            obs_array = self._obs_buffer
            # Ensure buffer is clear
            obs_array.fill(0)
        elif isinstance(observation, np.ndarray):
            # Use provided array
            obs_array = observation
        else:
            # Create a new array if incompatible type
            obs_array = np.zeros((obs_height, obs_width, self._feature_size), dtype=np.uint8)
        
        # Call the appropriate C++ implementation method
        if is_observer_id:
            self._cpp_mettagrid.observe(
                observer_id, 
                obs_width, 
                obs_height,
                <ObsType*>obs_array.data
            )
        else:
            self._cpp_mettagrid.observe_at(
                row, 
                col, 
                obs_width, 
                obs_height,
                <ObsType*>obs_array.data, 
                0  # last param is an ignored dummy uint8_t to help cython binding
            )
        
        # Return the observation only if we used our buffer or created a new one
        if observation is None or not isinstance(observation, np.ndarray):
            return obs_array


    def observe(self,
               GridObjectId observer_id,
               uint16_t obs_width,
               uint16_t obs_height,
               observation=None):
        """
        Get observation from a specific observer's perspective.
        
        Args:
            observer_id: ID of the observer object
            obs_width: Width of the observation window
            obs_height: Height of the observation window
            observation: Buffer to store the observation (numpy array)
        """
        return self._observe_internal(
            obs_width, 
            obs_height, 
            observation, 
            is_observer_id=True, 
            observer_id=observer_id
        )


    def observe_at(self,
                  uint16_t row,
                  uint16_t col,
                  uint16_t obs_width,
                  uint16_t obs_height,
                  observation=None):
        """
        Get observation at a specific location in the grid.
        
        Args:
            row: Row coordinate
            col: Column coordinate
            obs_width: Width of the observation window
            obs_height: Height of the observation window
            observation: Buffer to store the observation (numpy array)
        """
        return self._observe_internal(
            obs_width, 
            obs_height, 
            observation, 
            is_observer_id=False, 
            row=row, 
            col=col
        )
    
    
    def enable_reward_decay(self, int32_t decay_time_steps=-1):
        """Enable reward decay mechanism."""
        self._cpp_mettagrid.enable_reward_decay(decay_time_steps)
    
    
    def disable_reward_decay(self):
        """Disable reward decay mechanism."""
        self._cpp_mettagrid.disable_reward_decay()
    
    
    def get_episode_rewards(self):
        """Get the episode rewards."""
        return self._episode_rewards_np
    
    
    def get_episode_stats(self):
        """Get statistics from the game and agents."""
        stats_json = self._cpp_mettagrid.get_episode_stats_json()
        return json.loads(stats_json)


    def render_ascii(self):
        """Render the grid as an ASCII representation."""
        ascii_str = self._cpp_mettagrid.render_ascii()
        # Convert to numpy array for backward compatibility
        lines = ascii_str.strip().split('\n')
        grid = np.array([list(line) for line in lines], dtype=np.str_)
        return grid


    def render(self):
        """Render the grid to the console."""
        ascii_str = self._cpp_mettagrid.render_ascii()
        print(ascii_str, end='')  # end='' because string already has newlines
    
    
    # Gym compatibility properties
    @property
    def action_space(self):
        """Get the action space for gymnasium compatibility."""
        # Get number of action handlers and max argument value
        cdef:
            vector[uint8_t] max_args = self._cpp_mettagrid.max_action_args() 
            uint8_t* data_ptr = max_args.data()
            size_t size = max_args.size()
            uint8_t[:] max_args_view = <uint8_t[:size]>data_ptr
            cnp.ndarray max_args_array
            uint8_t max_arg
        
        max_args_array = np.asarray(max_args_view)
        
        # Use numpy's max function
        max_arg = max_args_array.max() if max_args_array.size > 0 else 0
        
        return gym.spaces.MultiDiscrete(
            [len(max_args_array), max_arg + 1],
            dtype=np.int64
        )
    
    
    @property
    def observation_space(self):
        """Get the observation space for gymnasium compatibility."""
        return gym.spaces.Box(
            0,
            255,
            shape=(self._cpp_mettagrid.map_height(), self._cpp_mettagrid.map_width(), self._feature_size),
            dtype=obs_np_type
        )