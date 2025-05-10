# Python and Cython imports
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int8_t, int32_t

import json
import numpy as np
cimport numpy as cnp
import gymnasium as gym
from omegaconf import DictConfig, ListConfig, OmegaConf

# Import types from the pxd file
from mettagrid.py_mettagrid cimport (
    CppMettaGrid, 
    GridObjectId, 
    c_observations_type,
    c_terminals_type,
    c_truncations_type,
    c_rewards_type,
    c_actions_type,
    c_masks_type,
    c_success_type,
    np_observations_type,
    np_terminals_type,
    np_truncations_type,
    np_rewards_type,
    np_actions_type,
    np_masks_type,
    np_success_type,
    ObjectTypeNames,
    InventoryItemNames
)

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
        cnp.ndarray _flattened_actions_buffer
        cnp.ndarray _obs_buffer
        uint32_t _num_agents


    def __init__(self, env_cfg: DictConfig | ListConfig, cnp.ndarray np_map):
        cdef:
            object cfg
            uint32_t num_agents
            uint32_t max_timestep
            uint16_t obs_width
            uint16_t obs_height
            uint16_t map_width
            uint16_t map_height
            string cfg_json_bytes
            string map_json_bytes
            str cfg_json
            str map_json
            
        # Initialize configuration
        cfg = OmegaConf.create(env_cfg.game)
        self._cfg = cfg
        
        # Extract parameters
        num_agents = cfg.num_agents
        max_timestep = cfg.max_steps
        obs_width = cfg.obs_width
        obs_height = cfg.obs_height
        map_width = np_map.shape[1]
        map_height = np_map.shape[0]
        
        self._num_agents = num_agents
        self._obs_width = obs_width
        self._obs_height = obs_height
        
        # Create the C++ MettaGrid instance with ownership of the grid
        self._cpp_mettagrid = new CppMettaGrid(
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
        self._grid_features_size = len(self._grid_features_list)
        
        # Pre-allocate NumPy arrays for common operations
        self._flattened_actions_buffer = np.zeros((num_agents, 2), dtype=np_actions_type).flatten()
        self._obs_buffer = np.zeros((self._obs_width, self._obs_height, self._grid_features_size), dtype=np_observations_type)


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
        cnp.ndarray rewards) -> None:
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
            uint32_t num_agents
            uint16_t obs_height
            uint16_t obs_width
            uint32_t grid_features_size
            uint32_t dim_i
            bint shape_match
            tuple expected_obs_shape
            tuple obs_shape_tuple
            tuple term_shape
            tuple trunc_shape
            tuple reward_shape
            cnp.ndarray[c_observations_type, ndim=4] typed_observations
            cnp.ndarray[c_terminals_type, ndim=1] typed_terminals
            cnp.ndarray[c_truncations_type, ndim=1] typed_truncations
            cnp.ndarray[c_rewards_type, ndim=1] typed_rewards
        
        num_agents = self._num_agents
        obs_height = self._cfg.obs_height
        obs_width = self._cfg.obs_width
        grid_features_size = self._grid_features_size
        shape_match = True
        
        # check buffers
        observations = np.ascontiguousarray(observations, dtype=np_observations_type)
        terminals = np.ascontiguousarray(terminals, dtype=np_terminals_type)
        truncations = np.ascontiguousarray(truncations, dtype=np_truncations_type)
        rewards = np.ascontiguousarray(rewards, dtype=np_rewards_type)
        
        # Create typed views of the data for internal use
        typed_observations = observations
        typed_terminals = terminals
        typed_truncations = truncations
        typed_rewards = rewards
        
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
            <c_observations_type*>observations.data,
            <c_terminals_type*>terminals.data,
            <c_truncations_type*>truncations.data,
            <c_rewards_type*>rewards.data,
        )


    # Helper method to get grid features as Python list - caching the result
    cdef list _get_grid_features(self):
        cdef:
            vector[string] grid_features = self._cpp_mettagrid.grid_features()
            list result = []
            uint32_t i
            string feature
        
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
        """
        Take a step in the environment with the given actions.
        
        Args:
            actions: NumPy array of actions, shape (num_agents, 2)
                
        Returns:
            Tuple containing observations, rewards, terminals, truncations, and info dict
        """
        cdef:
            cnp.ndarray[c_actions_type, ndim=1] flat_actions

        flat_actions = np.ascontiguousarray(actions.flatten(), dtype=np_actions_type)

        self._cpp_mettagrid.step(<c_actions_type*>flat_actions.data)
        self._cpp_mettagrid.compute_group_rewards(<float*>self._rewards_np.data)

        return (self._observations_np, self._rewards_np, self._terminals_np, self._truncations_np, {})


    cpdef cnp.ndarray[c_success_type, ndim=1] action_success(self):
        """
        Get the action success information.

        Returns:
            NumPy array indicating success/failure of actions
        """
        cdef:
            vector[c_success_type] success
            size_t size
            uint32_t i
            cnp.ndarray success_array
            
        success = self._cpp_mettagrid.action_success()
        size = success.size()

        # Create a new numpy array
        success_array = np.zeros(size, dtype=np_success_type)

        # Manual copy - accessing vector<bool> elements individually
        for i in range(size):
            success_array[i] = success[i]

        # Return the array
        return success_array


    cpdef cnp.ndarray[uint8_t, ndim=1] max_action_args(self):
        """
        Get the maximum action arguments.
        
        Returns:
            NumPy array of maximum action arguments
        """
        cdef:
            vector[uint8_t] max_args
            uint8_t* data_ptr
            size_t size
            uint32_t i
            uint8_t* target_ptr
            cnp.ndarray max_args_array
            
        max_args = self._cpp_mettagrid.max_action_args()
        data_ptr = max_args.data()
        size = max_args.size()
        
        # Create a new numpy array and manually copy the data
        max_args_array = np.zeros(size, dtype=np.uint8)
        
        # Manual copy
        if size > 0:
            target_ptr = <uint8_t*>max_args_array.data
            for i in range(size):
                target_ptr[i] = data_ptr[i]
        
        # Return the array
        return max_args_array


    cpdef uint32_t current_timestep(self):
        return self._cpp_mettagrid.current_timestep()
    
    
    cpdef uint16_t map_width(self):
        return self._cpp_mettagrid.map_width()
    
    
    cpdef uint16_t map_height(self):
        return self._cpp_mettagrid.map_height()
    
    
    cpdef list grid_features(self):
        return self._grid_features_list
    
    
    cpdef uint32_t num_agents(self):
        return self._num_agents


    cdef cnp.ndarray _observe_internal(self,
                         uint16_t obs_width,
                         uint16_t obs_height,
                         object observation,
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
            
        Returns:
            NumPy array containing the observation
        """
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
            obs_array = np.zeros((obs_width, obs_height, self._grid_features_size), dtype=np.uint8)
        
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
            obs_array = np.zeros((obs_width, obs_height, self._grid_features_size), dtype=np.uint8)
        
        # Call the appropriate C++ implementation method
        if is_observer_id:
            self._cpp_mettagrid.observe(
                observer_id, 
                obs_width, 
                obs_height,
                <c_observations_type*>obs_array.data
            )
        else:
            self._cpp_mettagrid.observe_at(
                row, 
                col, 
                obs_width, 
                obs_height,
                <c_observations_type*>obs_array.data, 
                0  # last param is an ignored dummy uint8_t to help cython binding
            )
        
        # Return the observation only if we used our buffer or created a new one
        if observation is None or not isinstance(observation, np.ndarray):
            return obs_array
        return None


    cpdef cnp.ndarray observe(self,
               GridObjectId observer_id,
               uint16_t obs_width,
               uint16_t obs_height,
               object observation=None):
        """
        Get observation from a specific observer's perspective.
        
        Args:
            observer_id: ID of the observer object
            obs_width: Width of the observation window
            obs_height: Height of the observation window
            observation: Buffer to store the observation (numpy array)
            
        Returns:
            NumPy array containing the observation
        """
        # Pass all required parameters explicitly
        return self._observe_internal(
            obs_width=obs_width,
            obs_height=obs_height,
            observation=observation,
            is_observer_id=True,
            observer_id=observer_id,
            row=0,  # Not used when is_observer_id is True, but must be provided
            col=0   # Not used when is_observer_id is True, but must be provided
        )


    cpdef cnp.ndarray observe_at(self,
                  uint16_t row,
                  uint16_t col,
                  uint16_t obs_width,
                  uint16_t obs_height,
                  object observation=None):
        """
        Get observation at a specific location in the grid.
        
        Args:
            row: Row coordinate
            col: Column coordinate
            obs_width: Width of the observation window
            obs_height: Height of the observation window
            observation: Buffer to store the observation (numpy array)
            
        Returns:
            NumPy array containing the observation
        """

        # Pass all required parameters explicitly
        return self._observe_internal(
            obs_width=obs_width,
            obs_height=obs_height,
            observation=observation,
            is_observer_id=False,
            observer_id=0,  # Not used when is_observer_id is False, but must be provided
            row=row,
            col=col
        )
    
    
    cpdef void enable_reward_decay(self, int32_t decay_time_steps=-1):
        self._cpp_mettagrid.enable_reward_decay(decay_time_steps)
    
    
    cpdef void disable_reward_decay(self):  
        self._cpp_mettagrid.disable_reward_decay()
    
    
    cpdef cnp.ndarray get_episode_rewards(self):
        """
        Get the episode rewards.
        
        Returns:
            NumPy array of episode rewards
        """
        cdef:
            float* episode_rewards_ptr
            uint32_t episode_rewards_size

        episode_rewards_ptr = self._cpp_mettagrid.get_episode_rewards()
        episode_rewards_size = self._cpp_mettagrid.get_episode_rewards_size()
        return np.asarray(<float[:episode_rewards_size]>episode_rewards_ptr)
    
    
    cpdef dict get_episode_stats(self):
        stats_json = self._cpp_mettagrid.get_episode_stats_json()
        return json.loads(stats_json)

    
    # Gym compatibility properties
    @property
    def action_space(self):
        """
        Get the action space for gymnasium compatibility.
        
        Returns:
            Gymnasium action space
        """
        cdef:
            vector[uint8_t] max_args
            uint8_t* data_ptr
            size_t size
            uint32_t i
            uint8_t* target_ptr
            cnp.ndarray max_args_array
            uint8_t max_arg

        max_args = self._cpp_mettagrid.max_action_args()
        data_ptr = max_args.data()
        size = max_args.size()
        
        # Create a new numpy array and manually copy the data
        max_args_array = np.zeros(size, dtype=np.uint8)
        
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
        """
        Get the observation space for gymnasium compatibility.
        
        Returns:
            Gymnasium observation space
        """
        return gym.spaces.Box(
            0,
            255,
            shape=(self.obs_height, self.obs_width, self._grid_features_size),
            dtype=np_observations_type
        )  # note that gym wants height, width which is opposite of our convention


    cpdef list object_type_names(self):
        """
        Get a list of all object type names.
        
        Returns:
            List of object type names
        """
        cdef:
            vector[string] cpp_names
            list result
            uint32_t i
            string name
            
        cpp_names = ObjectTypeNames
        result = []
        
        # Convert C++ vector of strings to Python list
        for i in range(cpp_names.size()):
            name = cpp_names[i]
            if isinstance(name, bytes):
                result.append(name.decode('utf8'))
            else:
                result.append(name)
        
        return result


    cpdef list inventory_item_names(self):
        """
        Get a list of all inventory item names.
        
        Returns:
            List of inventory item names
        """
        cdef:
            vector[string] cpp_names
            list result
            uint32_t i
            string name
            
        cpp_names = InventoryItemNames
        result = []
        
        # Convert C++ vector of strings to Python list
        for i in range(cpp_names.size()):
            name = cpp_names[i]
            if isinstance(name, bytes):
                result.append(name.decode('utf8'))
            else:
                result.append(name)
        
        return result


    cpdef list action_names(self):
        """
        Get a list of all action handler names.
        
        Returns:
            List of action handler names
        """
        cdef:
            vector[string] cpp_names
            list result
            uint32_t i
            string name
            
        result = []
           
        cpp_names = self._cpp_mettagrid.action_names()
        
        # Convert C++ vector of strings to Python list
        for i in range(cpp_names.size()):
            name = cpp_names[i]
            if isinstance(name, bytes):
                result.append(name.decode('utf8'))
            else:
                result.append(name)
        
        return result


    cpdef dict grid_objects(self):
        """
        Get information about all grid objects.
        
        Returns:
            A dictionary mapping object IDs to their properties.
        """
        cdef:
            string json_str
            dict objects_dict
        

        # Get JSON string from C++ and parse it
        json_str = self._cpp_mettagrid.get_grid_objects_json()
        objects_dict = json.loads(json_str)
        
        # Convert string keys to integers
        return {int(k): v for k, v in objects_dict.items()}


    @property
    def grid_features_list(self) -> list:
        return self._grid_features_list


    @property
    def grid_features_size(self) -> uint32_t:
        return self._grid_features_size


    @property
    def obs_width(self) -> uint16_t:
        return self._obs_width


    @property
    def obs_height(self) -> uint16_t:
        return self._obs_height

    @classmethod
    def get_numpy_type_name(cls, str type_id):
        """
        Python wrapper for the C++ get_numpy_type_name function.
        Returns the NumPy dtype name for a given type identifier.
        """
        # Convert Python string to bytes object - don't use temporary reference
        py_bytes = type_id.encode('utf8')  # Changed from type_id_str to type_id
        
        # Pass the bytes to the C function
        result = cls.get_numpy_type_name(<const char*>py_bytes)
        
        # Convert result back to Python string
        return result.decode('utf8')