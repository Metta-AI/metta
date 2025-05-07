# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int8_t, int32_t
from libcpp cimport bool

# Basic type definitions
ctypedef uint32_t GridObjectId
ctypedef uint32_t ActionType
ctypedef uint8_t ObsType

# Forward declarations - use extern to avoid redefinition
cdef extern from "grid_object.hpp":
    cdef struct GridLocation:
        uint32_t r
        uint32_t c
        uint32_t layer
        
        GridLocation(uint32_t row, uint32_t col, uint32_t l)

cdef extern from "action_handler.hpp":
    cdef cppclass ActionArg:
        int32_t value
        ActionArg(int32_t v)

# Forward declare the classes from other modules
cdef extern from "grid.hpp":
    cdef cppclass Grid
    
cdef extern from "grid_object.hpp":
    cdef cppclass GridObject
    
cdef extern from "event.hpp":
    cdef cppclass EventManager
    
cdef extern from "stats_tracker.hpp":
    cdef cppclass StatsTracker
    
cdef extern from "action_handler.hpp":
    cdef cppclass ActionHandler
    
cdef extern from "observation_encoder.hpp":
    cdef cppclass ObservationEncoder
    
cdef extern from "objects/agent.hpp":
    cdef cppclass Agent

# Updated MettaGrid class definition with precise types
cdef extern from "core.hpp":
    cdef cppclass MettaGrid:
        # Constructor - updated signature
        MettaGrid(uint32_t map_width, 
                 uint32_t map_height,
                 uint32_t num_agents, 
                 uint32_t max_timestep, 
                 uint16_t obs_width, 
                 uint16_t obs_height)
        
        # Core methods
        void init_action_handlers(const vector[ActionHandler*]& action_handlers)
        void add_agent(Agent* agent)
        void initialize_from_json(const string& map_json, const string& config_json)
        void reset()
        void step(int32_t** actions)
        
        # Observation methods
        void compute_observation(uint16_t observer_r, 
                               uint16_t observer_c,
                               uint16_t obs_width, 
                               uint16_t obs_height,
                               ObsType* observation)
        void compute_observations(int32_t** actions)
        void observe(GridObjectId observer_id, 
                    uint16_t obs_width,
                    uint16_t obs_height, 
                    ObsType* observation)
        void observe_at(uint16_t row, 
                       uint16_t col,
                       uint16_t obs_width, 
                       uint16_t obs_height,
                       ObsType* observation)

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
        
        # Getters - now returning const references to vectors
        const vector[ObsType]& get_observations() const
        const vector[int8_t]& get_terminals() const
        const vector[int8_t]& get_truncations() const
        const vector[float]& get_rewards() const
        const vector[float]& get_episode_rewards() const
        const vector[double]& get_group_rewards() const
        
        # Status and environment information
        uint32_t current_timestep() const
        uint32_t map_width() const
        uint32_t map_height() const
        vector[string] grid_features() const
        uint32_t num_agents() const
        vector[int8_t] action_success() const
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
        string render_ascii() const