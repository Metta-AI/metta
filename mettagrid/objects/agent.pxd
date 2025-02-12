# distutils: language=c++
# cython: warn.undeclared=False
# cython: c_api_binop_methods=True

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libc.stdio cimport printf
from mettagrid.observation_encoder cimport ObservationEncoder, ObsType
from mettagrid.grid_object cimport GridObject, TypeId, GridCoord, GridLocation, GridObjectId
from mettagrid.event cimport EventHandler, EventArg
from mettagrid.grid_agent cimport GridAgent
from mettagrid.objects.metta_object cimport ObjectConfig, GridLayer

cdef enum InventoryItem:
    r1 = 0,
    r2 = 1,
    r3 = 2,
    InventoryCount = 3

cdef vector[string] InventoryItemNames # defined in inventory.pyx

cdef cppclass Agent(GridAgent):
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

    Agent(
        GridCoord r, GridCoord c,
        string group_name,
        unsigned char group_id,
        ObjectConfig cfg)

    void update_inventory(InventoryItem item, short amount, float *reward)
    short update_energy(short amount, float *reward)

    inline void obs(ObsType[:] obs)

