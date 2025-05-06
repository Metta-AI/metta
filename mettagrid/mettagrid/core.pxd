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

# Only keep the MettaGrid class definition
cdef extern from "core.hpp":
    cdef cppclass MettaGrid:
        # Constructor
        MettaGrid(Grid* grid, unsigned int num_agents, unsigned int max_timestep, 
                 unsigned short obs_width, unsigned short obs_height)
        
        # Core methods
        void init_action_handlers(vector[ActionHandler*] action_handlers)
        void add_agent(Agent* agent)
        void compute_observation(unsigned observer_r, unsigned int observer_c,
                               unsigned short obs_width, unsigned short obs_height,
                               ObsType* observation)
        void compute_observations(int** actions)
        void step(int** actions)
        
        # Getters
        unsigned int current_timestep()
        unsigned int map_width()
        unsigned int map_height()
        vector[string] grid_features()
        unsigned int num_agents()
        
        # Reward decay methods
        void enable_reward_decay(int decay_time_steps)
        void disable_reward_decay()
        
        # Observation methods
        void observe(GridObjectId observer_id, unsigned short obs_width,
                    unsigned short obs_height, ObsType* observation)
        void observe_at(unsigned short row, unsigned short col,
                       unsigned short obs_width, unsigned short obs_height,
                       ObsType* observation)
                       
        # Accessors
        float* get_episode_rewards()
        vector[bool] action_success()
        vector[unsigned char] max_action_args()

        # Buffer management
        void set_buffers(ObsType* observations,
                  char* terminals,
                  char* truncations,
                  float* rewards,
                  float* episode_rewards,
                  unsigned int num_agents)
        
        # Group rewards
        void init_group_rewards(double* group_rewards, unsigned int size)
        void compute_group_rewards(float* rewards)
        void set_group_reward_pct(unsigned int group_id, float pct)
        void set_group_size(unsigned int group_id, unsigned int size)
        
        # Event management
        void init_event_manager(EventManager* event_manager)
        EventManager* get_event_manager()

        # Stats tracking
        StatsTracker* stats()
        void set_stats(StatsTracker* s)
        
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