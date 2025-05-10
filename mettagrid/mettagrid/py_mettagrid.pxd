# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int8_t, int32_t



# Forward declarations - updated to match our new structure
cdef extern from "types.hpp":
    ctypedef uint32_t GridObjectId
    ctypedef uint8_t ObsType
    ctypedef uint8_t ActionsType;
    ctypedef uint8_t numpy_bool_t;

    cdef struct Event:
        uint32_t timestamp
        uint16_t event_id
        GridObjectId object_id
        int32_t arg
        
        bint operator_lt "operator<"(const Event& other)

cdef extern from "event_manager.hpp":
    cdef cppclass EventHandler:
        EventHandler(EventManager* em)
        void handle_event(GridObjectId object_id, int32_t arg)
    
    cdef cppclass EventManager:
        void schedule_event(uint16_t event_id, uint32_t delay, GridObjectId object_id, int32_t arg)
        void process_events(uint32_t current_timestep)

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
        void init_action_handlers(const vector[ActionHandler*]& action_handlers)
        void add_agent(Agent* agent)
        void initialize_from_json(const string& map_json, const string& config_json)
        void reset()

        void step(ActionsType* flat_actions);
        
        
        # Observation methods
        void compute_observations(ActionsType* flat_actions);
                
        void compute_observation(uint16_t observer_r, 
                               uint16_t observer_c,
                               uint16_t obs_width, 
                               uint16_t obs_height,
                               ObsType* observation)

        void observe(GridObjectId observer_id, 
                    uint16_t obs_width,
                    uint16_t obs_height, 
                    ObsType* observation)
        void observe_at(uint16_t row, 
                       uint16_t col,
                       uint16_t obs_width, 
                       uint16_t obs_height,
                       ObsType* observation,
                       uint8_t dummy)

        # Observation utilities
        void observation_at(ObsType* flat_buffer,
                      uint32_t obs_width,
                      uint32_t obs_height,
                      uint32_t feature_size,
                      uint32_t r,
                      uint32_t c,
                      ObsType* output)
                      
        void set_observation_at(ObsType* flat_buffer,
                          uint32_t obs_width,
                          uint32_t obs_height,
                          uint32_t feature_size,
                          uint32_t r,
                          uint32_t c,
                          const ObsType* values)
        
        
        # Set external buffers method
        void set_buffers(ObsType* external_observations,
                       numpy_bool_t* external_terminals,
                       numpy_bool_t* external_truncations,
                       float* external_rewards)
        
        # Replace vector getters with pointer getters
        ObsType* get_observations() const
        numpy_bool_t* get_terminals() const
        numpy_bool_t* get_truncations() const
        float* get_rewards() const
        float* get_episode_rewards() const
        float* get_group_rewards() const
        
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
        vector[numpy_bool_t] action_success() const
        vector[uint8_t] max_action_args() const
        const vector[Agent*]& get_agents() const
        
        # Reward management
        void enable_reward_decay(int32_t decay_time_steps)
        void disable_reward_decay()
        void compute_group_rewards(float* rewards)
        void set_group_reward_pct(uint32_t group_id, float pct)
        void set_group_size(uint32_t group_id, uint32_t size)
        
        # Stats and management
        StatsTracker* stats() const
        EventManager* get_event_manager()
        string get_episode_stats_json() const

        vector[string] action_names() const
        string get_grid_objects_json() const
