from libc.stdio cimport printf

from mettagrid.event cimport Event, EventArg, EventHandler, EventId
from mettagrid.grid_env cimport GridEnv
from mettagrid.grid_object cimport GridObjectId

cdef class EventManager:
    def __init__(self, GridEnv env, list[EventHandler] event_handlers):
        self.env = env

        self._event_handlers = event_handlers
        for idx in range(len(event_handlers)):
            (<EventHandler>event_handlers[idx]).init(env, idx)
        self._current_timestep = 0

    cdef void schedule_event(
        self, EventId event_id, unsigned int delay,
        GridObjectId object_id, EventArg arg):

        cdef Event event = Event(
            self._current_timestep + delay,
            event_id,
            object_id,
            arg
        )
        # printf("Scheduling event %d for timestep %d\n", event_id, event.timestamp)
        self._event_queue.push(event)

    cdef void process_events(self, unsigned int current_timestep):
        self._current_timestep = current_timestep
        cdef Event event
        while not self._event_queue.empty():
            event = self._event_queue.top()
            if event.timestamp > self._current_timestep:
                break
            self._event_queue.pop()
            (<EventHandler>self._event_handlers[event.event_id]).handle_event(event.object_id, event.arg)

cdef class EventHandler:
    cdef void init(self, GridEnv env, EventId event_id):
        self.env = env
        self.event_id = event_id

    cdef void schedule(self, unsigned int delay, GridObjectId object_id, EventArg arg):
        self.event_manager.schedule_event(self.event_id, delay, object_id, arg)

    cdef void handle_event(self, GridObjectId object_id, int arg):
        pass
