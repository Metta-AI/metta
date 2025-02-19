from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action cimport ActionArg
from mettagrid.actions.actions cimport MettaActionHandler
from mettagrid.grid_object cimport (
    GridLocation,
    GridObject,
    GridObjectId,
    Orientation
)
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.altar cimport Altar
from mettagrid.objects.armory cimport Armory
from mettagrid.objects.constants cimport Events, GridLayer, ObjectType
from mettagrid.objects.factory cimport Factory
from mettagrid.objects.generator cimport Generator
from mettagrid.objects.lab cimport Lab
from mettagrid.objects.lasery cimport Lasery
from mettagrid.objects.metta_object cimport MettaObject
from mettagrid.objects.mine cimport Mine
from mettagrid.objects.temple cimport Temple
from mettagrid.objects.usable cimport Usable

cdef class Use(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "use")

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
        if target == NULL or not target.is_usable_type():
            return False

        cdef Usable *usable = <Usable*> target

        if not usable.usable(actor):
            return False


        usable.ready = 0
        self.env._event_manager.schedule_event(Events.Reset, usable.cooldown, usable.id, 0)

        actor.stats.incr(self._stats.target[target._type_id])
        actor.stats.incr(self._stats.target[target._type_id], actor.group_name)
        actor.stats.set_once(self._stats.target_first_use[target._type_id], self.env._current_timestep)

        usable.use(actor, &self.env._rewards[actor_id])
        return True
