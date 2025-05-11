from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int8_t, int32_t
from libc.string cimport strcmp

import numpy as np

# Forward declarations - import directly from types.hpp
cdef extern from "types.hpp":
    # Core type definitions
    ctypedef uint32_t GridObjectId
    ctypedef uint8_t c_actions_type
    ctypedef uint8_t numpy_bool_t
    ctypedef uint8_t c_observations_type
    ctypedef numpy_bool_t c_terminals_type
    ctypedef numpy_bool_t c_truncations_type
    ctypedef float c_rewards_type
    ctypedef numpy_bool_t c_masks_type
    ctypedef numpy_bool_t c_success_type
    
    # Numpy type name constants
    const char* NUMPY_OBSERVATIONS_TYPE
    const char* NUMPY_TERMINALS_TYPE
    const char* NUMPY_TRUNCATIONS_TYPE
    const char* NUMPY_REWARDS_TYPE
    const char* NUMPY_ACTIONS_TYPE
    const char* NUMPY_MASKS_TYPE
    const char* NUMPY_SUCCESS_TYPE
    
    # Type mapping function
    const char* get_numpy_type_name(const char* type_id)
    
    # Struct definitions
    cdef struct GridLocation:
        uint32_t r
        uint32_t c
        uint32_t layer
        
        GridLocation(uint32_t row, uint32_t col, uint32_t l)
    
    cdef struct Event:
        uint32_t timestamp
        uint16_t event_id
        GridObjectId object_id
        int32_t arg
        
        bint operator_lt "operator<"(const Event& other)

# Import numpy types for C
np_observations_type = np.dtype(get_numpy_type_name("observations").decode('utf8'))
np_terminals_type = np.dtype(get_numpy_type_name("terminals").decode('utf8'))
np_truncations_type = np.dtype(get_numpy_type_name("truncations").decode('utf8'))
np_rewards_type = np.dtype(get_numpy_type_name("rewards").decode('utf8'))
np_actions_type = np.dtype(get_numpy_type_name("actions").decode('utf8'))
np_masks_type = np.dtype(get_numpy_type_name("masks").decode('utf8'))
np_success_type = np.dtype(get_numpy_type_name("success").decode('utf8'))

cdef extern from "event_manager.hpp":
    cdef cppclass EventHandler:
        EventHandler(EventManager* em)
        void handle_event(GridObjectId object_id, int32_t arg) except +
    
    cdef cppclass EventManager:
        void schedule_event(uint16_t event_id, uint32_t delay, GridObjectId object_id, int32_t arg) except +
        void process_events(uint32_t current_timestep) except +

cdef extern from "grid_object.hpp":
    cdef struct GridLocation:
        uint32_t r
        uint32_t c
        uint32_t layer
        
        GridLocation(uint32_t row, uint32_t col, uint32_t l)
    
    cdef cppclass GridObject

cdef extern from "actions/action_handler.hpp":
    cdef cppclass ActionHandler 

cdef extern from "grid.hpp":
    cdef cppclass Grid
    
cdef extern from "stats_tracker.hpp":
    cdef cppclass StatsTracker
    
cdef extern from "constants.hpp":
    cdef vector[string] ObjectTypeNames
    cdef vector[string] InventoryItemNames

cdef extern from "objects/agent.hpp":
    cdef cppclass Agent

# Updated MettaGrid class definition with precise types
cdef extern from "core.hpp":
    cdef cppclass CppMettaGrid:
        # Constructor - updated signature
        CppMettaGrid(
            uint32_t map_width, 
            uint32_t map_height,
            uint32_t num_agents, 
            uint32_t max_timestep, 
            uint16_t obs_width, 
            uint16_t obs_height
        )
        
        # Core methods
        void init_action_handlers(const vector[ActionHandler*]& action_handlers) except +
        void add_agent(Agent* agent) except +
        void initialize_from_json(const string& map_json, const string& config_json) except +
        void reset() except +

        void step(c_actions_type* flat_actions) except +
        
        
        # Observation methods
        void compute_observations(c_actions_type* flat_actions) except +
                
        void compute_observation(uint16_t observer_r, 
                               uint16_t observer_c,
                               uint16_t obs_width, 
                               uint16_t obs_height,
                               c_observations_type* observation) except +

        void observe(GridObjectId observer_id, 
                    uint16_t obs_width,
                    uint16_t obs_height, 
                    c_observations_type* observation) except +

        void observe_at(uint16_t row, 
                       uint16_t col,
                       uint16_t obs_width, 
                       uint16_t obs_height,
                       c_observations_type* observation,
                       uint8_t dummy) except +

        # Observation utilities
        void observation_at(c_observations_type* flat_buffer,
                      uint32_t obs_width,
                      uint32_t obs_height,
                      uint32_t feature_size,
                      uint32_t r,
                      uint32_t c,
                      c_observations_type* output) except +
                      
        void set_observation_at(c_observations_type* flat_buffer,
                          uint32_t obs_width,
                          uint32_t obs_height,
                          uint32_t feature_size,
                          uint32_t r,
                          uint32_t c,
                          const c_observations_type* values) except +
        
        # Set external buffers method
        void set_buffers(c_observations_type* external_observations,
                       numpy_bool_t* external_terminals,
                       numpy_bool_t* external_truncations,
                       float* external_rewards) except +
        
        # Replace vector getters with pointer getters
        c_observations_type* get_observations() const
        c_terminals_type* get_terminals() const
        c_truncations_type* get_truncations() const
        c_rewards_type* get_rewards() const
        c_rewards_type* get_episode_rewards() const
        c_rewards_type* get_group_rewards() const
        
        # Size getters
        size_t get_observations_size() const
        size_t get_terminals_size() const
        size_t get_truncations_size() const
        size_t get_rewards_size() const
        size_t get_episode_rewards_size() const
        size_t get_group_rewards_size() const
        
        # Status and environment information
        uint32_t current_timestep() const
        uint32_t map_width() const
        uint32_t map_height() const
        vector[string] grid_features() const
        uint32_t num_agents() const
        vector[c_success_type] action_success() const
        vector[uint8_t] max_action_args() const
        const vector[Agent*]& get_agents() const
        
        # Reward management
        void enable_reward_decay(int32_t decay_time_steps) except +
        void disable_reward_decay() except +
        void compute_group_rewards(c_rewards_type* rewards) except +
        void set_group_reward_pct(uint32_t group_id, float pct) except +
        void set_group_size(uint32_t group_id, uint32_t size) except +
        
        # Stats and management
        StatsTracker* stats() const
        EventManager* get_event_manager() except +
        string get_episode_stats_json() const

        vector[string] action_names() const
        string get_grid_objects_json() const
