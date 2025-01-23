from libcpp.queue cimport priority_queue

from mettagrid.grid_object cimport GridObjectId
from mettagrid.grid_env cimport GridEnv

cdef extern from "event.hpp":
    ctypedef unsigned short EventId
    ctypedef int EventArg
    cdef struct Event:
        unsigned int timestamp
        EventId event_id
        GridObjectId object_id
        EventArg arg


cdef class EventManager:
    cdef:
        GridEnv env
        priority_queue[Event] _event_queue
        unsigned int _current_timestep
        list[EventHandler] _event_handlers

    cdef void schedule_event(
        self,
        EventId event_id,
        unsigned int delay,
        GridObjectId object_id,
        EventArg arg)

    cdef void process_events(self, unsigned int current_timestep)

cdef class EventHandler:
    cdef GridEnv env
    cdef EventId event_id

    cdef void init(self, GridEnv env, EventId event_id)
    cdef void schedule(self, unsigned int delay, GridObjectId object_id, EventArg arg)
    cdef void handle_event(self, GridObjectId object_id, int arg)
