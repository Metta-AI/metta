from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from mettagrid.stats_tracker cimport StatsTracker
from .constants cimport InventoryItem
from .metta_object cimport MettaObject, ObjectConfig

cdef extern from "agent.hpp":
    cdef cppclass Agent(MettaObject):
        unsigned char group
        unsigned char group
        unsigned char frozen
        unsigned char freeze_duration
        unsigned char orientation
        vector[unsigned char] inventory
        unsigned char max_items
        vector[float] resource_rewards
        float action_failure_penalty
        string group_name
        unsigned char color
        unsigned char agent_id
        StatsTracker stats
        # This should reference the reward buffer in the GridEnv, which
        # accumulates rewards for all agents on a per-step basis.
        float *reward

        Agent(GridCoord r, GridCoord c,
            string group_name,
            unsigned char group_id,
            ObjectConfig cfg,
            map[string, float] rewards)

        void update_inventory(InventoryItem item, short amount)

        # This is used to link the agent's reward to the grid's rewards.
        # It's valid to call this multiple times, if you need to re-link the
        # reward.
        void init(float *reward)

        @staticmethod
        inline vector[string] feature_names()
