
from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from mettagrid.action cimport ActionArg
from mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from mettagrid.objects cimport Generator, InventoryItem, ObjectTypeNames, InventoryItemNames
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

        usable.ready = 0
        self.env._event_manager.schedule_event(Events.Reset, usable.cooldown, usable.id, 0)

        actor.stats.incr(self._stats.target[target._type_id])
        actor.stats.incr(self._stats.target[target._type_id], actor.group_name)
        actor.stats.set_once(self._stats.target_first_use[target._type_id], self.env._current_timestep)

        actor.stats.add(self._stats.target_energy[target._type_id], usable.use_cost + self.action_cost)
        actor.stats.add(self._stats.target_energy[target._type_id], actor.group_name, usable.use_cost + self.action_cost)

        usable.use(actor, actor_id, &self.env._rewards[actor_id])

        return True
