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

        Agent(GridCoord r, GridCoord c,
            string group_name,
            unsigned char group_id,
            ObjectConfig cfg,
            map[string, float] rewards)

        void update_inventory(InventoryItem item, short amount, float *reward)

        @staticmethod
        inline vector[string] feature_names()
