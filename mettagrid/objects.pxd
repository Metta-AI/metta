# distutils: language=c++
# cython: warn.undeclared=False
# cython: c_api_binop_methods=True

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libc.stdio cimport printf
from mettagrid.stats_tracker cimport StatsTracker
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
        return this.ready

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        pass

cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    MineT = 2
    GeneratorT = 3
    AltarT = 4
    ArmoryT = 5
    LaseryT = 6
    LabT = 7
    FactoryT = 8
    TempleT = 9
    Count = 10

cdef vector[string] ObjectTypeNames # defined in objects.pyx

cdef enum InventoryItem:
    ore = 0,
    battery = 1,
    heart = 2,
    armor = 3,
    laser = 4,
    blueprint = 5,
    InventoryCount = 6

cdef vector[string] InventoryItemNames # defined in objects.pyx

cdef cppclass Agent(MettaObject):
    unsigned char group
    unsigned char frozen
    unsigned char freeze_duration
    unsigned char orientation
    vector[unsigned char] inventory
    unsigned char max_items
    vector[float] resource_rewards
    float freeze_reward
    string group_name
    unsigned char color
    unsigned char agent_id
    StatsTracker stats

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
        this.freeze_duration = cfg[b"freeze_duration"]
        this.orientation = 0
        this.inventory.resize(InventoryItem.InventoryCount)
        this.max_items = cfg[b"max_inventory"]
        this.resource_rewards.resize(InventoryItem.InventoryCount)
        this.resource_rewards[InventoryItem.ore] = 0
        this.resource_rewards[InventoryItem.battery] = 0.01
        this.resource_rewards[InventoryItem.heart] = 1
        this.resource_rewards[InventoryItem.armor] = 0
        this.resource_rewards[InventoryItem.laser] = 0
        this.resource_rewards[InventoryItem.blueprint] = 0
        this.freeze_reward = float(cfg[b"freeze_reward"]) / 1000.0
        this.color = 0

    inline void update_inventory(InventoryItem item, short amount, float *reward):
        this.inventory[<InventoryItem>item] += amount
        reward[0] += amount * this.resource_rewards[<InventoryItem>item]

        if this.inventory[<InventoryItem>item] > this.max_items:
            this.inventory[<InventoryItem>item] = this.max_items

        if amount > 0:
            this.stats.add(InventoryItemNames[item], b"gained", amount)
            this.stats.add(InventoryItemNames[item], b"gained", this.group_name, amount)
        else:
            this.stats.add(InventoryItemNames[item], b"lost", -amount)
            this.stats.add(InventoryItemNames[item], b"lost", this.group_name, -amount)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.group
        obs[2] = this.hp
        obs[3] = this.frozen
        obs[4] = this.orientation
        obs[5] = this.color
        cdef unsigned short idx = 6

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
            "agent:orientation",
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

cdef cppclass Mine(Usable):

    inline Mine(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.MineT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline bint usable(const Agent *actor):
        return Usable.usable(actor)

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        actor.update_inventory(InventoryItem.ore, 1, rewards)
        actor.stats.incr(InventoryItemNames[InventoryItem.ore], b"created")

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
        return ["mine", "mine:hp", "mine:ready"]


cdef cppclass Generator(Usable):
    inline Generator(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.GeneratorT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline bint usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[InventoryItem.ore] > 0

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        actor.update_inventory(InventoryItem.ore, -1, rewards)
        actor.update_inventory(InventoryItem.battery, 1, rewards)

        actor.stats.incr(InventoryItemNames[InventoryItem.ore], b"used")
        actor.stats.incr(
            InventoryItemNames[InventoryItem.ore],
            b"converted",
            InventoryItemNames[InventoryItem.battery])

        actor.stats.incr(InventoryItemNames[InventoryItem.battery], b"created")

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
        return ["generator", "generator:hp", "generator:ready"]

cdef cppclass Altar(Usable):
    inline Altar(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.AltarT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline bint usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[InventoryItem.battery] > 2

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        actor.update_inventory(InventoryItem.battery, -3, rewards)
        actor.update_inventory(InventoryItem.heart, 1, rewards)

        actor.stats.add(InventoryItemNames[InventoryItem.battery], b"used", 3)
        actor.stats.incr(InventoryItemNames[InventoryItem.heart], b"created")
        actor.stats.add(
            InventoryItemNames[InventoryItem.battery],
            b"converted",
            InventoryItemNames[InventoryItem.heart], 3)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["altar", "altar:hp", "altar:ready"]

cdef cppclass Armory(Usable):
    inline Armory(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.ArmoryT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline bint usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[InventoryItem.ore] > 2

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        actor.update_inventory(InventoryItem.ore, -3, rewards)
        actor.update_inventory(InventoryItem.armor, 1, rewards)

        actor.stats.add(InventoryItemNames[InventoryItem.ore], b"used", 3)
        actor.stats.incr(InventoryItemNames[InventoryItem.armor], b"created")

        actor.stats.add(
            InventoryItemNames[InventoryItem.ore],
            b"converted",
            InventoryItemNames[InventoryItem.armor], 3)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
        return ["armory", "armory:hp", "armory:ready"]

cdef cppclass Lasery(Usable):
    inline Lasery(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.LaseryT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline bint usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[InventoryItem.ore] > 0 and actor.inventory[InventoryItem.battery] > 1

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        actor.update_inventory(InventoryItem.ore, -1, rewards)
        actor.update_inventory(InventoryItem.battery, -2, rewards)
        actor.update_inventory(InventoryItem.laser, 1, rewards)

        actor.stats.add(InventoryItemNames[InventoryItem.ore], b"used", 1)
        actor.stats.add(InventoryItemNames[InventoryItem.battery], b"used", 2)
        actor.stats.incr(InventoryItemNames[InventoryItem.laser], b"created")
        actor.stats.add(
            InventoryItemNames[InventoryItem.ore],
            b"converted",
            InventoryItemNames[InventoryItem.laser], 1)
        actor.stats.add(
            InventoryItemNames[InventoryItem.battery],
            b"converted",
            InventoryItemNames[InventoryItem.laser], 2)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
            return ["lasery", "lasery:hp", "lasery:ready"]

cdef cppclass Lab(Usable):
    inline Lab(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.LabT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline bint usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[InventoryItem.battery] > 2 and actor.inventory[InventoryItem.ore] > 2

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        actor.update_inventory(InventoryItem.battery, -3, rewards)
        actor.update_inventory(InventoryItem.ore, -3, rewards)
        actor.update_inventory(InventoryItem.blueprint, 1, rewards)

        actor.stats.add(InventoryItemNames[InventoryItem.battery], b"used", 3)
        actor.stats.add(InventoryItemNames[InventoryItem.ore], b"used", 3)
        actor.stats.incr(InventoryItemNames[InventoryItem.blueprint], b"created")

        actor.stats.add(
            InventoryItemNames[InventoryItem.battery],
            b"converted",
            InventoryItemNames[InventoryItem.blueprint], 3)
        actor.stats.add(
            InventoryItemNames[InventoryItem.ore],
            b"converted",
            InventoryItemNames[InventoryItem.blueprint], 3)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
        return ["lab", "lab:hp", "lab:ready"]

cdef cppclass Factory(Usable):
    inline Factory(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.FactoryT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline bint usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[InventoryItem.blueprint] > 0 and actor.inventory[InventoryItem.ore] > 4 and actor.inventory[InventoryItem.battery] > 4

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        actor.update_inventory(InventoryItem.blueprint, -1, rewards)
        actor.update_inventory(InventoryItem.ore, -5, rewards)
        actor.update_inventory(InventoryItem.battery, -5, rewards)
        actor.update_inventory(InventoryItem.armor, 5, rewards)
        actor.update_inventory(InventoryItem.laser, 5, rewards)

        actor.stats.add(InventoryItemNames[InventoryItem.blueprint], b"used", 1)
        actor.stats.add(InventoryItemNames[InventoryItem.armor], b"created", 5)
        actor.stats.add(InventoryItemNames[InventoryItem.laser], b"created", 5)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
        return ["factory", "factory:hp", "factory:ready"]

cdef cppclass Temple(Usable):
    inline Temple(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.TempleT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline bint usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[InventoryItem.heart] > 0 and actor.inventory[InventoryItem.blueprint] > 0

    inline void use(Agent *actor, unsigned int actor_id, float *rewards):
        actor.update_inventory(InventoryItem.heart, -1, rewards)
        actor.update_inventory(InventoryItem.blueprint, -1, rewards)
        actor.update_inventory(InventoryItem.heart, 5, rewards)

        actor.stats.add(InventoryItemNames[InventoryItem.heart], b"used", 1)
        actor.stats.add(InventoryItemNames[InventoryItem.blueprint], b"used", 1)
        actor.stats.add(InventoryItemNames[InventoryItem.heart], b"created", 5)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.ready

    @staticmethod
    inline vector[string] feature_names():
        return ["temple", "temple:hp", "temple:ready"]

cdef class ResetHandler(EventHandler):
    cdef inline void handle_event(self, GridObjectId obj_id, EventArg arg):
        cdef Usable *usable = <Usable*>self.env._grid.object(obj_id)
        if usable is NULL:
            return

        usable.ready = True
        self.env._stats.incr(b"resets", ObjectTypeNames[usable._type_id])

cdef enum Events:
    Reset = 0

cdef map[TypeId, GridLayer] ObjectLayers
