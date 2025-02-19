from libc.stdio cimport printf
from libcpp.string cimport string

from omegaconf import OmegaConf

from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.grid_object cimport GridObjectId
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.constants cimport ObjectTypeNames

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class MettaActionHandler(ActionHandler):
    def __init__(self, cfg: OmegaConf, action_name: str):
        ActionHandler.__init__(self, action_name)

        self._stats.success = "action." + action_name
        self._stats.failure = "action." + action_name + ".failed"
        self._stats.first_use = "action." + action_name + ".first_use"

        for t, n in enumerate(ObjectTypeNames):
            self._stats.target[t] = self._stats.success + "." + n
            self._stats.target_first_use[t] = self._stats.first_use + "." + n

    cdef bint handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef Agent *actor = <Agent*>self.env._grid.object(actor_object_id)

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
            self.env._rewards[actor_id] -= actor.action_failure_penalty
            actor.stats.set_once(self._stats.first_use, self.env._current_timestep)

        return result

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False
