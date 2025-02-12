# distutils: language=c++
# cython: warn.undeclared=False
# cython: c_api_binop_methods=True

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libc.stdio cimport printf
from mettagrid.observation_encoder cimport ObservationEncoder, ObsType
from mettagrid.grid_object cimport GridObject, TypeId, GridCoord, GridLocation, GridObjectId


cdef cppclass Converter(Usable):
    short prey_r1_output_energy;
    short predator_r1_output_energy;
    short predator_r2_output_energy;

    inline Converter(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.ConverterT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)
        this.prey_r1_output_energy = cfg[b"energy_output.r1.prey"]
        this.predator_r1_output_energy = cfg[b"energy_output.r1.predator"]
        this.predator_r2_output_energy = cfg[b"energy_output.r2.predator"]

    inline bint usable(const Agent *actor):
        return Usable.usable(actor) and (
            actor.inventory[InventoryItem.r1] > 0 or
            (actor.inventory[InventoryItem.r2] > 0 and
            actor.group_name == b"predator")
        )

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        cdef unsigned int energy_gain = 0
        cdef InventoryItem consumed_resource = InventoryItem.r1
        cdef InventoryItem produced_resource = InventoryItem.r2
        cdef char stat_name[256]
        cdef unsigned int potential_energy_gain = this.prey_r1_output_energy
        if actor.group_name == b"predator":
            if actor.inventory[InventoryItem.r2] > 0:
                # eat meat if you can
                consumed_resource = InventoryItem.r2
                produced_resource = InventoryItem.r3
                potential_energy_gain = this.predator_r2_output_energy
            else:
                potential_energy_gain = this.predator_r1_output_energy
                produced_resource = InventoryItem.r3

        actor.update_inventory(consumed_resource, -1, NULL)
        actor.stats.incr(InventoryItemNames[consumed_resource], b"used", actor.group_name)

        actor.update_inventory(produced_resource, 1, NULL)
        actor.stats.incr(InventoryItemNames[produced_resource], b"gained", actor.group_name)

        energy_gain = actor.update_energy(potential_energy_gain, rewards)
        actor.stats.add(b"energy.gained", energy_gain, actor.group_name)

    inline obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
        return ["converter", "converter:hp", "converter:ready"]
