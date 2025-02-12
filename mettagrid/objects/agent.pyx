# distutils: language=c++

from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
from mettagrid.grid_object cimport GridObject, GridObjectId
from mettagrid.objects.metta_object cimport MettaObject

cdef vector[string] InventoryItemNames = <vector[string]>[
    "r1",
    "r2",
    "r3"
]

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
from mettagrid.objects.metta_object cimport ObjectConfig, GridLayer

cdef enum InventoryItem:
    r1 = 0,
    r2 = 1,
    r3 = 2,
    InventoryCount = 3

cdef vector[string] InventoryItemNames # defined in inventory.pyx

cdef cppclass Agent(MettaObject):
    inline Agent(
        GridCoord r, GridCoord c,
        string group_name,
        unsigned char group_id,
        ObjectConfig cfg):
        GridObject.init(ObjectType.AgentT, GridLocation(r, c, GridLayer.Agent_Layer))

        this.group_name = group_name
        this.group = group_id
        this.frozen = 0
        this.attack_damage = cfg[b"attack_damage"]
        this.freeze_duration = cfg[b"freeze_duration"]
        this.max_energy = cfg[b"max_energy"]
        this.energy = 0
        this.update_energy(cfg[b"initial_energy"], NULL)
        this.shield_upkeep = cfg[b"upkeep.shield"]
        this.orientation = 0
        this.inventory.resize(InventoryItem.InventoryCount)
        this.max_items = cfg[b"max_inventory"]
        this.energy_reward = float(cfg[b"energy_reward"]) / 1000.0
        this.resource_reward = float(cfg[b"resource_reward"]) / 1000.0
        this.freeze_reward = float(cfg[b"freeze_reward"]) / 1000.0
        this.shield = False
        this.color = 0

    inline void update_inventory(InventoryItem item, short amount, float *reward):
        this.inventory[<InventoryItem>item] += amount
        if reward is not NULL and amount > 0:
            reward[0] += amount * this.resource_reward

        if this.inventory[<InventoryItem>item] > this.max_items:
            this.inventory[<InventoryItem>item] = this.max_items

    inline short update_energy(short amount, float *reward):
        if amount < 0:
            amount = max(-this.energy, amount)
        else:
            amount = min(this.max_energy - this.energy, amount)

        this.energy += amount
        if reward is not NULL and amount > 0:
            reward[0] += amount * this.energy_reward

        return amount

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.group
        obs[2] = this.frozen
        obs[3] = this.energy
        obs[4] = this.orientation
        obs[5] = this.shield
        obs[6] = this.color
        cdef unsigned short idx = 7

        cdef unsigned short i
        for i in range(InventoryItem.InventoryCount):
            obs[idx + i] = this.inventory[i]

    @staticmethod
    inline vector[string] feature_names():
        return [
            "agent",
            "agent:group",
            "agent:hp",
            "agent:frozen",
            "agent:energy",
            "agent:orientation",
            "agent:shield",
            "agent:color"
        ] + [
            "agent:inv:" + n for n in InventoryItemNames]
