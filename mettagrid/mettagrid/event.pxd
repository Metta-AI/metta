from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from mettagrid.grid_object cimport GridObjectId
from mettagrid.grid cimport Grid
from mettagrid.stats_tracker cimport StatsTracker

cdef extern from "event.hpp":
    ctypedef unsigned short EventId
    ctypedef int EventArg
    cdef struct Event:
        unsigned int timestamp
        EventId event_id
        GridObjectId object_id
        EventArg arg

    cdef cppclass EventManager:
        Grid *grid
        StatsTracker *stats
        priority_queue[Event] event_queue
        unsigned int current_timestep
        vector[EventHandler*] event_handlers

        EventManager()

        void init(Grid *grid, StatsTracker *stats)

        void schedule_event(
            EventId event_id,
            unsigned int delay,
            GridObjectId object_id,
            EventArg arg)

        void process_events(unsigned int current_timestep)

    cdef cppclass EventHandler:
        EventManager *event_manager

        EventHandler(EventManager *event_manager)

        void handle_event(GridObjectId object_id, EventArg arg)
