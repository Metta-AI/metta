
from libc.stdio cimport printf
from libc.string cimport strcat, strcpy

from omegaconf import OmegaConf

from mettagrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from mettagrid.actions.actions cimport MettaActionHandler

cdef class Use(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "use")

        self._stats.first_use = b"action.use.first_use"

        for t, n in enumerate(ObjectTypeNames):
            self._stats.target_first_use[t] = self._stats.first_use + "." + n

    cdef unsigned char max_arg(self):
        return 0

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef char stat_name[256]
        cdef GridLocation target_loc = self.env._grid.relative_location(
            actor.location,
            <Orientation>actor.orientation
        )
        target_loc.layer = GridLayer.Object_Layer
        cdef MettaObject *target = <MettaObject*>self.env._grid.object_at(target_loc)
        if target == NULL:
            return False

        if not target.usable(actor):
            return False

        cdef Usable *usable = <Usable*> target
        actor.update_energy(-usable.use_cost, &self.env._rewards[actor_id])

        usable.ready = 0
        self.env._event_manager.schedule_event(Events.Reset, usable.cooldown, usable.id, 0)

        self.env._stats.agent_incr(actor_id, self._stats.target[target._type_id].c_str())
        strcpy(stat_name, actor.group_name.c_str())
        strcat(stat_name, ".")
        strcat(stat_name, self._stats.target[target._type_id].c_str())
        self.env._stats.agent_incr(actor_id, stat_name)
        self.env._stats.agent_set_once(actor_id, self._stats.target_first_use[target._type_id].c_str(), self.env._current_timestep)
        strcpy(stat_name, actor.group_name.c_str())
        strcat(stat_name, ".")
        strcat(stat_name, self._stats.target_energy[target._type_id].c_str())
        self.env._stats.agent_add(actor_id, stat_name, usable.use_cost + self.action_cost)

        if target._type_id == ObjectType.AltarT:
            self.env._rewards[actor_id] += 1

        cdef Generator *generator
        if target._type_id == ObjectType.GeneratorT:
            generator = <Generator*>target
            generator.r1 -= 1
            actor.update_inventory(InventoryItem.r1, 1)
            self.env._stats.agent_incr(actor_id, "r1.gained")
            strcpy(stat_name, actor.group_name.c_str())
            strcat(stat_name, ".r1.gained")
            self.env._stats.agent_incr(actor_id, stat_name)
            self.env._stats.game_incr("r1.harvested")

        cdef Converter *converter
        if target._type_id == ObjectType.ConverterT:
            converter = <Converter*>target
            converter.use(actor, actor_id, self.env._stats, &self.env._rewards[actor_id])

        return True
