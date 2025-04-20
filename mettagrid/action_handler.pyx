from mettagrid.action_handler cimport ActionArg
from mettagrid.grid cimport Grid
from mettagrid.grid_object cimport GridObjectId
from mettagrid.objects.constants cimport ObjectTypeNames
from mettagrid.objects.agent cimport Agent

cdef class ActionHandler:
    def __init__(self, string action_name):
        self._action_name = action_name
        self._priority = 0

        self._stats.success = "action." + action_name
        self._stats.failure = "action." + action_name + ".failed"
        self._stats.first_use = "action." + action_name + ".first_use"

        for t, n in enumerate(ObjectTypeNames):
            self._stats.target[t] = self._stats.success + "." + n
            self._stats.target_first_use[t] = self._stats.first_use + "." + n

    cdef void init(self, Grid *grid):
        self._grid = grid

    cdef bint handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg,
        unsigned int current_timestep):

        cdef Agent *actor = <Agent*>self._grid.object(actor_object_id)

        if actor.frozen > 0:
            actor.stats.incr(b"status.frozen.ticks")
            actor.stats.incr(b"status.frozen.ticks", actor.group_name)
            actor.frozen -= 1
            return False

        cdef bint result = self._handle_action(actor_id, actor, arg)

        if result:
            actor.stats.incr(self._stats.success)
        else:
            actor.stats.incr(self._stats.failure)
            actor.stats.incr(b"action.failure_penalty")
            actor.reward[0] -= actor.action_failure_penalty
            actor.stats.set_once(self._stats.first_use, current_timestep)

        return result

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False

    cdef unsigned char max_arg(self):
        return 0

    cpdef string action_name(self):
        return self._action_name
