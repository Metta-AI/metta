from mettagrid.event cimport EventHandler, EventArg
from mettagrid.grid_object cimport GridObjectId
from .constants cimport ObjectTypeNames, Events
from .converter cimport Converter

cdef class ProductionHandler(EventHandler):
    cdef inline void handle_event(self, GridObjectId obj_id, EventArg arg):
        cdef Converter *converter = <Converter*>self.env._grid.object(obj_id)
        if converter is NULL:
            return
        
        converter.finish_converting()
        self.env._stats.incr(b"finished_production", ObjectTypeNames[converter._type_id])

        if converter.maybe_start_converting():
            self.env._event_manager.schedule_event(Events.FinishConverting, converter.recipe_duration, converter.id, 0)