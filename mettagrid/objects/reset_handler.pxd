from mettagrid.event cimport EventHandler, EventArg
from mettagrid.grid_object cimport GridObjectId
from .constants cimport ObjectTypeNames
from .usable cimport Usable

cdef class ResetHandler(EventHandler):
    cdef inline void handle_event(self, GridObjectId obj_id, EventArg arg):
        cdef Usable *usable = <Usable*>self.env._grid.object(obj_id)
        if usable is NULL:
            return

        usable.ready = True
        self.env._stats.incr(b"resets", ObjectTypeNames[usable._type_id])