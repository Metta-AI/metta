# distutils: language=c++
# cython: warn.undeclared=False
# cython: c_api_binop_methods=True

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from mettagrid.grid_env import StatsTracker
from libc.stdio cimport printf
from mettagrid.observation_encoder cimport ObservationEncoder, ObsType
from mettagrid.grid_object cimport GridObject, TypeId, GridCoord, GridLocation, GridObjectId
from mettagrid.event cimport EventHandler, EventArg

cdef enum GridLayer:
    Agent_Layer = 0
    Object_Layer = 1

ctypedef map[string, int] ObjectConfig

cdef cppclass MettaObject(GridObject):
    unsigned int hp

    inline void init_mo(ObjectConfig cfg):
        this.hp = cfg[b"hp"]

    inline bint usable(const Agent *actor):
        return False

cdef cppclass Usable(MettaObject):
    unsigned int use_cost
    unsigned int cooldown
    unsigned char ready

    inline void init_usable(ObjectConfig cfg):
        this.use_cost = cfg[b"use_cost"]
        this.cooldown = cfg[b"cooldown"]
        this.ready = 1

    inline bint usable(const Agent *actor):
        return this.ready and this.use_cost <= actor.energy

cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    GeneratorT = 2
    ConverterT = 3
    AltarT = 4
    Count = 5

cdef vector[string] ObjectTypeNames # defined in objects.pyx

cdef enum InventoryItem:
    r1 = 0,
    r2 = 1,
    r3 = 2,
    InventoryCount = 3

cdef vector[string] InventoryItemNames # defined in objects.pyx


cdef cppclass Agent(MettaObject):
    unsigned char species
    unsigned char frozen
    unsigned char freeze_duration
    unsigned char energy
    unsigned char orientation
    unsigned char shield
    unsigned char shield_upkeep
    vector[unsigned char] inventory
    unsigned char max_items
    unsigned char max_energy
    float energy_reward

    inline Agent(GridCoord r, GridCoord c, unsigned char species, ObjectConfig cfg):
        GridObject.init(ObjectType.AgentT, GridLocation(r, c, GridLayer.Agent_Layer))
        MettaObject.init_mo(cfg)
        this.species = species
        this.frozen = 0
        this.freeze_duration = cfg[b"freeze_duration"]
        this.max_energy = cfg[b"max_energy"]
        this.energy = 0
        this.update_energy(cfg[b"initial_energy"], NULL)
        this.shield_upkeep = cfg[b"upkeep.shield"]
        this.orientation = 0
        this.inventory.resize(InventoryItem.InventoryCount)
        this.max_items = cfg[b"max_inventory"]
        this.energy_reward = float(cfg[b"energy_reward"]) / 1000.0
        this.shield = False

    inline void update_inventory(InventoryItem item, short amount):
        this.inventory[<InventoryItem>item] += amount
        if this.inventory[<InventoryItem>item] > this.max_items:
            this.inventory[<InventoryItem>item] = this.max_items

    inline short update_energy(short amount, float *reward):
        if amount < 0:
            amount = max(-this.energy, amount)
        else:
            amount = min(this.max_energy - this.energy, amount)

        this.energy += amount
        if reward is not NULL:
            reward[0] += amount * this.energy_reward

        return amount

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.species
        obs[2] = this.hp
        obs[3] = this.frozen
        obs[4] = this.energy
        obs[5] = this.orientation
        obs[6] = this.shield

        cdef unsigned short idx = 6
        cdef unsigned short i
        for i in range(InventoryItem.InventoryCount):
            obs[idx + i] = this.inventory[i]

    @staticmethod
    inline vector[string] feature_names():
        return [
            "agent",
            "agent:species",
            "agent:hp",
            "agent:frozen",
            "agent:energy",
            "agent:orientation",
            "agent:shield"
        ] + [
            "agent:inv:" + n for n in InventoryItemNames]

cdef cppclass Wall(MettaObject):
    inline Wall(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.WallT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]

cdef cppclass Generator(Usable):
    unsigned int r1

    inline Generator(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.GeneratorT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)
        this.r1 = cfg[b"initial_resources"]

    inline bint usable(const Agent *actor):
        # Only prey (0) can use generators.
        return Usable.usable(actor) and this.r1 > 0 and actor.species == 0

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.r1
        obs[3] = this.ready and this.r1 > 0


    @staticmethod
    inline vector[string] feature_names():
        return ["generator", "generator:hp", "generator:r1", "generator:ready"]

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
            actor.species == 1 and actor.inventory[InventoryItem.r2] > 0
        )

    inline void use(Agent *actor, unsigned int actor_id, StatsTracker *stats, float *rewards):
        cdef unsigned int energy_gain = 0
        cdef InventoryItem consumed_resource = InventoryItem.r1
        cdef InventoryItem produced_resource = InventoryItem.r2
        cdef unsigned int potential_energy_gain = this.prey_r1_output_energy
        if actor.species == 1:
            if actor.inventory[InventoryItem.r2] > 0:
                # eat meat if you can
                consumed_resource = InventoryItem.r2
                produced_resource = InventoryItem.r3
                potential_energy_gain = this.predator_r2_output_energy
            else:
                potential_energy_gain = this.predator_r1_output_energy
        
        actor.update_inventory(consumed_resource, -1)
        stats[0].agent_incr(actor_id, InventoryItemNames[consumed_resource] + ".used")
        stats[0].agent_incr(actor_id, "." + actor.species_name + "." + InventoryItemNames[consumed_resource] + ".used")

        actor.update_inventory(produced_resource, 1)
        stats[0].agent_incr(actor_id, InventoryItemNames[produced_resource] + ".gained")
        stats[0].agent_incr(actor_id, "." + actor.species_name + "." + InventoryItemNames[produced_resource] + ".gained")

        energy_gain = actor.update_energy(potential_energy_gain, rewards)
        stats[0].agent_add(actor_id, "energy.gained", energy_gain)
        stats[0].agent_add(actor_id, "." + actor.species_name + ".energy.gained", energy_gain)

    inline obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready
        obs[3] = this.prey_r1_output_energy
        obs[4] = this.predator_r1_output_energy
        obs[5] = this.predator_r2_output_energy

    @staticmethod
    inline vector[string] feature_names():
        return ["converter", "converter:hp", "converter:ready", "converter:prey_r1_output_energy", "converter:predator_r1_output_energy", "converter:predator_r2_output_energy"]

cdef cppclass Altar(Usable):
    inline Altar(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.AltarT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["altar", "altar:hp", "altar:ready"]

cdef map[TypeId, GridLayer] ObjectLayers

cdef class ResetHandler(EventHandler):
    cdef inline void handle_event(self, GridObjectId obj_id, EventArg arg):
        cdef Usable *usable = <Usable*>self.env._grid.object(obj_id)
        if usable is NULL:
            return

        usable.ready = True
        self.env._stats.game_incr("resets." + ObjectTypeNames[usable._type_id])

cdef enum Events:
    Reset = 0
