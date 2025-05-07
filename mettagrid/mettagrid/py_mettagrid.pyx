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
from mettagrid.py_mettagrid cimport MettaGrid, GridObjectId, ObsType, ObjectTypeNames, InventoryItemNames

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
        
        # observation dimensions
        uint16_t _obs_width
        uint16_t _obs_height

        # Cache for frequently accessed data
        list _grid_features_list
        uint32_t _grid_features_size
        
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
        self._obs_width = cfg.obs_width
        self._obs_height = cfg.obs_height
        map_width = np_map.shape[1]
        map_height = np_map.shape[0]
        
        # Create the C++ MettaGrid instance with ownership of the grid
        self._cpp_mettagrid = new MettaGrid(
            map_width, map_height, num_agents, max_timestep, self._obs_width, self._obs_height
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
        self._grid_features_size = len(self._grid_features_list)
        
        # Set up NumPy array views for Python access
        self._create_numpy_views(num_agents, self._grid_features_size,  self._obs_width, self._obs_height)
        
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
        self._obs_buffer = np.zeros((self._obs_width, self._obs_height, self._grid_features_size), dtype=np.uint8)


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

    def _create_numpy_views(self, uint32_t num_agents, uint32_t grid_features_size, uint16_t obs_width, uint16_t obs_height):
        """Create NumPy array views that reference the C++ internal buffers."""
        # Declare all variables at the beginning of the function
        cdef:
            # For observations
            vector[ObsType] cpp_observations = self._cpp_mettagrid.get_observations()
            ObsType* obs_ptr = cpp_observations.data()
            size_t obs_size = cpp_observations.size()
            
            # For terminals
            vector[int8_t] cpp_terminals = self._cpp_mettagrid.get_terminals()
            int8_t* term_ptr = cpp_terminals.data()
            size_t term_size = cpp_terminals.size()
            
            # For truncations
            vector[int8_t] cpp_truncations = self._cpp_mettagrid.get_truncations()
            int8_t* trunc_ptr = cpp_truncations.data()
            size_t trunc_size = cpp_truncations.size()
            
            # For rewards
            vector[float] cpp_rewards = self._cpp_mettagrid.get_rewards()
            float* reward_ptr = cpp_rewards.data()
            size_t reward_size = cpp_rewards.size()
            
            # For episode rewards
            vector[float] cpp_episode_rewards = self._cpp_mettagrid.get_episode_rewards()
            float* ep_reward_ptr = cpp_episode_rewards.data()
            size_t ep_reward_size = cpp_episode_rewards.size()
            
            # For group rewards
            vector[double] cpp_group_rewards = self._cpp_mettagrid.get_group_rewards()
            double* group_reward_ptr = cpp_group_rewards.data()
            size_t group_reward_size = cpp_group_rewards.size()
            
            # Additional variables for manual copying
            size_t expected_size
            size_t copy_size
            uint32_t i
            uint8_t* target_ptr
            int8_t* term_target_ptr
            int8_t* trunc_target_ptr
            float* reward_target_ptr
            float* ep_reward_target_ptr
            double* group_reward_target_ptr
            
            # NumPy arrays
            cnp.ndarray obs_array
            cnp.ndarray terminals_array
            cnp.ndarray truncations_array
            cnp.ndarray rewards_array
            cnp.ndarray episode_rewards_array
            cnp.ndarray group_rewards_array
        
        # Create observations array
        obs_array = np.zeros((num_agents, obs_height, obs_width, grid_features_size), dtype=np.uint8)
        
        # Manually copy data if available
        if obs_size > 0:
            # We know exactly what size the array should be
            expected_size = num_agents * obs_height * obs_width * grid_features_size
            
            # Safety check to avoid buffer overruns
            copy_size = min(obs_size, expected_size)
            
            # Get raw pointers for direct memory access
            target_ptr = <uint8_t*>obs_array.data
            
            # Copy data manually
            for i in range(copy_size):
                target_ptr[i] = obs_ptr[i]
        
        # Create and copy terminals array
        terminals_array = np.zeros(num_agents, dtype=np.int8)
        if term_size > 0:
            copy_size = min(term_size, num_agents)
            term_target_ptr = <int8_t*>terminals_array.data
            for i in range(copy_size):
                term_target_ptr[i] = term_ptr[i]
        
        # Create and copy truncations array  
        truncations_array = np.zeros(num_agents, dtype=np.int8)
        if trunc_size > 0:
            copy_size = min(trunc_size, num_agents)
            trunc_target_ptr = <int8_t*>truncations_array.data
            for i in range(copy_size):
                trunc_target_ptr[i] = trunc_ptr[i]
        
        # Create and copy rewards array
        rewards_array = np.zeros(num_agents, dtype=np.float32)
        if reward_size > 0:
            copy_size = min(reward_size, num_agents)
            reward_target_ptr = <float*>rewards_array.data
            for i in range(copy_size):
                reward_target_ptr[i] = reward_ptr[i]
        
        # Create and copy episode rewards array
        episode_rewards_array = np.zeros(num_agents, dtype=np.float32)
        if ep_reward_size > 0:
            copy_size = min(ep_reward_size, num_agents)
            ep_reward_target_ptr = <float*>episode_rewards_array.data
            for i in range(copy_size):
                ep_reward_target_ptr[i] = ep_reward_ptr[i]
        
        # Create and copy group rewards array
        group_rewards_array = np.zeros(num_agents, dtype=np.float64)
        if group_reward_size > 0:
            copy_size = min(group_reward_size, num_agents)
            group_reward_target_ptr = <double*>group_rewards_array.data
            for i in range(copy_size):
                group_reward_target_ptr[i] = group_reward_ptr[i]
        
        # Assign arrays to instance variables
        self._observations_np = obs_array
        self._terminals_np = terminals_array
        self._truncations_np = truncations_array
        self._rewards_np = rewards_array
        self._episode_rewards_np = episode_rewards_array
        self._group_rewards_np = group_rewards_array

    def set_buffers(self, 
                    cnp.ndarray observations,
                    cnp.ndarray terminals, 
                    cnp.ndarray truncations, 
                    cnp.ndarray rewards):
        """
        Set external buffers for observations, terminals, truncations, and rewards.
        
        This allows sharing memory between the environment and the caller.
        
        Args:
            observations: NumPy array for observations
            terminals: NumPy array for terminal flags
            truncations: NumPy array for truncation flags
            rewards: NumPy array for rewards
        """
        cdef:
            uint32_t num_agents = self._num_agents
            uint16_t obs_height = self._cfg.obs_height
            uint16_t obs_width = self._cfg.obs_width
            uint32_t grid_features_size = self._grid_features_size
            uint32_t dim_i
            bint shape_match = True
            tuple expected_obs_shape
            tuple obs_shape_tuple
            tuple term_shape
            tuple trunc_shape
            tuple reward_shape
        
        # Predict expected buffer shapes
        expected_obs_shape = (num_agents, obs_height, obs_width, grid_features_size)
        
        # Convert NumPy shapes to Python tuples for comparison and error messages
        obs_shape_tuple = tuple(observations.shape[i] for i in range(observations.ndim))
        term_shape = tuple(terminals.shape[i] for i in range(terminals.ndim))
        trunc_shape = tuple(truncations.shape[i] for i in range(truncations.ndim))
        reward_shape = tuple(rewards.shape[i] for i in range(rewards.ndim))
        
        # Validate observation shape
        if observations.ndim != 4 or obs_shape_tuple != expected_obs_shape:
            raise ValueError(f"Observations buffer has shape {obs_shape_tuple}, expected {expected_obs_shape}")
        
        # Validate other buffer shapes
        if terminals.ndim < 1 or terminals.shape[0] < num_agents:
            raise ValueError(f"Terminals buffer has shape {term_shape}, expected first dimension ≥ {num_agents}")
        
        if truncations.ndim < 1 or truncations.shape[0] < num_agents:
            raise ValueError(f"Truncations buffer has shape {trunc_shape}, expected first dimension ≥ {num_agents}")
        
        if rewards.ndim < 1 or rewards.shape[0] < num_agents:
            raise ValueError(f"Rewards buffer has shape {reward_shape}, expected first dimension ≥ {num_agents}")
        
        # Store the external buffers
        self._observations_np = observations
        self._terminals_np = terminals
        self._truncations_np = truncations
        self._rewards_np = rewards


    # Helper method to get grid features as Python list - caching the result
    cdef list _get_grid_features(self):
        cdef vector[string] grid_features = self._cpp_mettagrid.grid_features()
        cdef list result = []
        cdef uint32_t i
        for i in range(grid_features.size()):
            # Check if the object is bytes before decoding
            feature = grid_features[i]
            if isinstance(feature, bytes):
                result.append(feature.decode('utf8'))
            else:
                # It's already a string, no need to decode
                result.append(feature)
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
            cnp.ndarray[int32_t, ndim=2] actions_array
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
            uint32_t i
            int8_t* target_ptr
            cnp.ndarray success_array
        
        # Create a new numpy array and manually copy the data
        success_array = np.zeros(size, dtype=np.int8)
        
        # Manual copy
        if size > 0:
            target_ptr = <int8_t*>success_array.data
            for i in range(size):
                target_ptr[i] = data_ptr[i]
        
        # Return the array
        return success_array

    def max_action_args(self):
        """Get the maximum action arguments."""
        cdef:
            vector[uint8_t] max_args = self._cpp_mettagrid.max_action_args() 
            uint8_t* data_ptr = max_args.data()
            size_t size = max_args.size()
            uint32_t i
            uint8_t* target_ptr
            cnp.ndarray max_args_array
        
        # Create a new numpy array and manually copy the data
        max_args_array = np.zeros(size, dtype=np.uint8)
        
        # Manual copy
        if size > 0:
            target_ptr = <uint8_t*>max_args_array.data
            for i in range(size):
                target_ptr[i] = data_ptr[i]
        
        # Return the array
        return max_args_array

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
            obs_array = self._obs_buffer # index as width, height
            # Ensure buffer is clear
            obs_array.fill(0)
        elif isinstance(observation, np.ndarray):
            # Use provided array
            obs_array = observation
        else:
            # Create a new array if incompatible type
            obs_array = np.zeros(( obs_width, obs_height, self._grid_features_size), dtype=np.uint8)
        
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
        cdef:
            vector[uint8_t] max_args = self._cpp_mettagrid.max_action_args() 
            uint8_t* data_ptr = max_args.data()
            size_t size = max_args.size()
            uint32_t i
            uint8_t* target_ptr
            cnp.ndarray max_args_array
            uint8_t max_arg
        
        # Create a new numpy array and manually copy the data
        max_args_array = np.zeros(size, dtype=np.uint8)
        
        # Manual copy
        if size > 0:
            target_ptr = <uint8_t*>max_args_array.data
            for i in range(size):
                target_ptr[i] = data_ptr[i]
        
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
            shape=(self.obs_height, self.obs_width, self._grid_features_size),
            dtype=obs_np_type
        )  # note that gym wants height, width which is opposite of our convention

    def object_type_names(self):
        """Get a list of all object type names."""
        cdef vector[string] cpp_names = ObjectTypeNames
        cdef list result = []
        cdef uint32_t i
        
        # Convert C++ vector of strings to Python list
        for i in range(cpp_names.size()):
            name = cpp_names[i]
            if isinstance(name, bytes):
                result.append(name.decode('utf8'))
            else:
                result.append(name)
        
        return result

    def inventory_item_names(self):
        """Get a list of all inventory item names."""
        cdef vector[string] cpp_names = InventoryItemNames
        cdef list result = []
        cdef uint32_t i
        
        # Convert C++ vector of strings to Python list
        for i in range(cpp_names.size()):
            name = cpp_names[i]
            if isinstance(name, bytes):
                result.append(name.decode('utf8'))
            else:
                result.append(name)
        
        return result

    def action_names(self):
        """Get a list of all action handler names."""
        cdef vector[string] cpp_names = self._cpp_mettagrid.action_names()
        cdef list result = []
        cdef uint32_t i
        
        # Convert C++ vector of strings to Python list
        for i in range(cpp_names.size()):
            name = cpp_names[i]
            if isinstance(name, bytes):
                result.append(name.decode('utf8'))
            else:
                result.append(name)
        
        return result

    def grid_objects(self):
        """
        Get information about all grid objects.
        
        Returns:
            A dictionary mapping object IDs to their properties.
        """
        # Get JSON string from C++ and parse it
        json_str = self._cpp_mettagrid.get_grid_objects_json()
        objects_dict = json.loads(json_str)
        
        # Convert string keys to integers
        return {int(k): v for k, v in objects_dict.items()}

    @property
    def grid_features_list(self):
        """Get the list of grid features."""
        return self._grid_features_list

    @property
    def grid_features_size(self):
        """Get the number of grid features."""
        return self._grid_features_size

    @property
    def observation_width(self):
        """Get the width of the observation window."""
        return self._obs_width

    @property
    def observation_height(self):
        """Get the height of the observation window."""
        return self._obs_height

            
