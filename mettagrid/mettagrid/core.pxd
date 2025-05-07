# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libc.stdint cimport uint8_t
from libcpp cimport bool

# Basic type definitions
ctypedef unsigned int GridObjectId
ctypedef unsigned int ActionType
ctypedef uint8_t ObsType

# Forward declarations - use extern to avoid redefinition
cdef extern from "grid_object.hpp":
    cdef struct GridLocation:
        unsigned int r
        unsigned int c
        unsigned int layer
        
        GridLocation(unsigned int row, unsigned int col, unsigned int l)

cdef extern from "action_handler.hpp":
    cdef cppclass ActionArg:
        int value
        ActionArg(int v)

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

# Updated MettaGrid class definition to match refactored C++ code
cdef extern from "core.hpp":
    cdef cppclass MettaGrid:
        # Constructor - updated signature
        MettaGrid(unsigned int map_width, 
                 unsigned int map_height,
                 unsigned int num_agents, 
                 unsigned int max_timestep, 
                 unsigned short obs_width, 
                 unsigned short obs_height)
        
        # Core methods
        void init_action_handlers(const vector[ActionHandler*]& action_handlers)
        void add_agent(Agent* agent)
        void initialize_from_json(const string& map_json, const string& config_json)
        void reset()
        void step(int** actions)
        
        # Observation methods
        void compute_observation(unsigned short observer_r, 
                               unsigned short observer_c,
                               unsigned short obs_width, 
                               unsigned short obs_height,
                               ObsType* observation)
        void compute_observations(int** actions)
        void observe(GridObjectId observer_id, 
                    unsigned short obs_width,
                    unsigned short obs_height, 
                    ObsType* observation)
        void observe_at(unsigned short row, 
                       unsigned short col,
                       unsigned short obs_width, 
                       unsigned short obs_height,
                       ObsType* observation)
                       
        # Observation utilities
        void observation_at(ObsType* flat_buffer,
                      unsigned int obs_width,
                      unsigned int obs_height,
                      unsigned int feature_size,
                      unsigned int r,
                      unsigned int c,
                      ObsType* output)
                      
        void set_observation_at(ObsType* flat_buffer,
                          unsigned int obs_width,
                          unsigned int obs_height,
                          unsigned int feature_size,
                          unsigned int r,
                          unsigned int c,
                          const ObsType* values)
        
        # Getters - now returning const references to vectors
        const vector[ObsType]& get_observations() const
        const vector[char]& get_terminals() const
        const vector[char]& get_truncations() const
        const vector[float]& get_rewards() const
        const vector[float]& get_episode_rewards() const
        const vector[double]& get_group_rewards() const
        
        # Status and environment information
        unsigned int current_timestep() const
        unsigned int map_width() const
        unsigned int map_height() const
        vector[string] grid_features() const
        unsigned int num_agents() const
        vector[char] action_success() const
        vector[unsigned char] max_action_args() const
        const vector[Agent*]& get_agents() const
        
        # Reward management
        void enable_reward_decay(int decay_time_steps)
        void disable_reward_decay()
        void compute_group_rewards(float* rewards)
        void set_group_reward_pct(unsigned int group_id, float pct)
        void set_group_size(unsigned int group_id, unsigned int size)
        
        # Stats and management
        StatsTracker* stats() const
        EventManager* get_event_manager()
        string get_episode_stats_json() const
        string render_ascii() const