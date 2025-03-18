from mettagrid.event cimport EventHandler, EventArg, EventManager
from mettagrid.grid_object cimport GridObjectId
from .constants cimport ObjectTypeNames, Events
from .converter cimport Converter

cdef extern from "production_handler.hpp":
    cdef cppclass ProductionHandler(EventHandler):
        ProductionHandler(EventManager *event_manager)

        void handle_event(GridObjectId obj_id, EventArg arg)

    cdef cppclass CoolDownHandler(EventHandler):
        CoolDownHandler(EventManager *event_manager)

        void handle_event(GridObjectId obj_id, EventArg arg)
