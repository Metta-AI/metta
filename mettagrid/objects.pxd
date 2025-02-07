# distutils: language=c++
# cython: warn.undeclared=False
# cython: c_api_binop_methods=True

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from mettagrid.grid_env cimport StatsTracker
from libc.stdio cimport printf
from mettagrid.observation_encoder cimport ObservationEncoder, ObsType
from mettagrid.grid_object cimport GridObject, TypeId, GridCoord, GridLocation, GridObjectId
from mettagrid.event cimport EventHandler, EventArg
from libc.string cimport strcat, strcpy
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
    string group_name
    unsigned char color

    inline Agent(
        GridCoord r, GridCoord c,
        string group_name,
        unsigned char group_id,
        ObjectConfig cfg):
        GridObject.init(ObjectType.AgentT, GridLocation(r, c, GridLayer.Agent_Layer))
        MettaObject.init_mo(cfg)

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
        this.shield = False
        this.color = 0

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
        obs[1] = this.group
        obs[2] = this.hp
        obs[3] = this.frozen
        obs[4] = this.energy
        obs[5] = this.orientation
        obs[6] = this.shield
        obs[7] = this.color
        cdef unsigned short idx = 8

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
        return Usable.usable(actor) and this.r1 > 0

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
            (actor.inventory[InventoryItem.r2] > 0 and
            actor.group_name == b"predator")
        )

    inline void use(Agent *actor, unsigned int actor_id, StatsTracker stats, float *rewards):
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

        actor.update_inventory(consumed_resource, -1)
        stats.agent_incr(actor_id, InventoryItemNames[consumed_resource] + ".used")
        strcpy(stat_name, actor.group_name.c_str())
        strcat(stat_name, ".")
        strcat(stat_name, InventoryItemNames[consumed_resource].c_str())
        strcat(stat_name, ".used")
        stats.agent_incr(actor_id, stat_name)

        actor.update_inventory(produced_resource, 1)
        stats.agent_incr(actor_id, InventoryItemNames[produced_resource] + ".gained")
        stats.agent_incr(actor_id, actor.group_name + "." + InventoryItemNames[produced_resource] + ".gained")

        energy_gain = actor.update_energy(potential_energy_gain, rewards)
        stats.agent_add(actor_id, "energy.gained", energy_gain)
        stats.agent_add(actor_id, actor.group_name + ".energy.gained", energy_gain)

    inline obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
        return ["converter", "converter:hp", "converter:ready"]

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
