from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation, GridObject
from mettagrid.stats_tracker cimport StatsTracker
from .constants cimport InventoryItem
from .metta_object cimport MettaObject, ObjectConfig

cdef extern from "agent.hpp":
    cdef cppclass Agent(MettaObject):
        unsigned char group
        unsigned char frozen
        unsigned char attack_damage
        unsigned char freeze_duration
        unsigned char energy
        unsigned char orientation
        unsigned char shield
        unsigned char shield_upkeep
        vector[unsigned char] inventory
        unsigned char max_items
        unsigned char max_energy
        float energy_reward
        float resource_reward
        float freeze_reward
        string group_name
        unsigned char color
        unsigned char agent_id
        StatsTracker stats

        Agent(GridCoord r, GridCoord c, string group_name, unsigned char group_id, ObjectConfig cfg)
        void update_inventory(InventoryItem item, short amount, float *reward)
        short update_energy(short amount, float *reward)
        @staticmethod
        vector[string] feature_names()
