
from libc.stdio cimport printf

from omegaconf import OmegaConf

from puffergrid.grid_object cimport GridObjectId
from puffergrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport Agent, ObjectTypeNames

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class MettaActionHandler(ActionHandler):
    def __init__(self, cfg: OmegaConf, action_name, action_cost=0):
        self.action_name = action_name

        self._stats.action = "action." + action_name
        self._stats.action_energy = "action." + action_name + ".energy"

        for t, n in enumerate(ObjectTypeNames):
            self._stats.target[t] = self._stats.action + "." + n
            self._stats.target_energy[t] = self._stats.action_energy + "." + n

        self.action_cost = cfg.cost

    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef Agent *actor = <Agent*>self.env._grid.object(actor_object_id)

        if actor.frozen:
            return False

        if actor.energy < self.action_cost:
            return False

        actor.energy -= self.action_cost
        self.env._stats.agent_add(actor_id, self._stats.action_energy.c_str(), self.action_cost)

        cdef char result = self._handle_action(actor_id, actor, arg)

        if result:
            self.env._stats.agent_incr(actor_id, self._stats.action.c_str())

        return result

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False




