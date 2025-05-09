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
from mettagrid.py_mettagrid cimport CppMettaGrid, GridObjectId, ObsType, ObjectTypeNames, InventoryItemNames

# Constants
obs_np_type = np.uint8

# Wrapper class for the C++ implementation
cdef class MettaGrid:
    cdef:
        object _cfg
        # The C++ implementation
        CppMettaGrid* _cpp_mettagrid
        
        # NumPy array views for Python access
        cnp.ndarray _observations_np
        cnp.ndarray _terminals_np
        cnp.ndarray _truncations_np
        cnp.ndarray _rewards_np
        
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
        self._cpp_mettagrid = new CppMettaGrid(
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
        

        # Pre-allocate NumPy arrays for common operations
        self._action_array_buffer = np.zeros((num_agents, 2), dtype=np.uint8)
        self._obs_buffer = np.zeros((self._obs_width, self._obs_height, self._grid_features_size), dtype=np.uint8)


    def __dealloc__(self):

        # Clean up the C++ object
        if self._cpp_mettagrid != NULL:
            del self._cpp_mettagrid
            self._cpp_mettagrid = NULL

    def set_buffers(
        self, 
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
        
        # check buffers
        observations = np.ascontiguousarray(observations, dtype=np.uint8)
        terminals = np.ascontiguousarray(terminals, dtype=np.int8)
        truncations = np.ascontiguousarray(truncations, dtype=np.int8)
        rewards = np.ascontiguousarray(rewards, dtype=np.float32)
        
        # Predict expected buffer shapes
        expected_obs_shape = (num_agents, obs_width, obs_height, grid_features_size)
        
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
        
        # Connect these arrays to the C++ engine
        self._cpp_mettagrid.set_buffers(
            <ObsType*>observations.data,
            <int8_t*>terminals.data,
            <int8_t*>truncations.data,
            <float*>rewards.data,
        )


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


    cpdef tuple[cnp.ndarray, dict] reset(self):
        self._cpp_mettagrid.reset()
        return (self._observations_np, {})


    cpdef tuple[cnp.ndarray, cnp.ndarray, cnp.ndarray, cnp.ndarray, dict] step(self, cnp.ndarray actions):
        """Take a step in the environment with the given actions."""
        cdef:
            tuple shape_tuple
            
        if actions.ndim != 2 or actions.shape[0] != self._num_agents or actions.shape[1] != 2:
            # Convert shape to a Python tuple
            shape_tuple = tuple([actions.shape[i] for i in range(actions.ndim)])
            raise ValueError("Actions must have shape ({0}, 2), got {1}".format(
                self._num_agents, shape_tuple))

        actions_flat = np.ascontiguousarray(actions, dtype=np.uint8).reshape(-1)
        self._cpp_mettagrid.step(actions_flat.tobytes())

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
    def obs_width(self):
        """Get the width of the observation window."""
        return self._obs_width

    @property
    def obs_height(self):
        """Get the height of the observation window."""
        return self._obs_height

            
