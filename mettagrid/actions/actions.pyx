
from libc.stdio cimport printf
from libcpp.string cimport string
from omegaconf import OmegaConf
from mettagrid.grid_object cimport GridObjectId
from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport Agent, ObjectTypeNames

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class MettaActionHandler(ActionHandler):
    def __init__(self, cfg: OmegaConf, action_name: str):
        ActionHandler.__init__(self, action_name)

        self._stats.action = "action." + action_name
        self._stats.action_energy = "action." + action_name + ".energy"

        for t, n in enumerate(ObjectTypeNames):
            self._stats.target[t] = self._stats.action + "." + n
            self._stats.target_energy[t] = self._stats.action_energy + "." + n

        self.action_cost = cfg.cost

    cdef bint handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef Agent *actor = <Agent*>self.env._grid.object(actor_object_id)
        cdef char stat_name[256]

        if actor.shield:
            actor.update_energy(-actor.shield_upkeep, &self.env._rewards[actor_id])
            actor.stats.add(b"shield_upkeep", actor.shield_upkeep)

            actor.stats.incr(b"shield_upkeep", actor.group_name)

            actor.stats.incr(b"status.shield.ticks")
            actor.stats.incr(b"status.shield.ticks", actor.group_name)

            if actor.energy == 0:
                actor.shield = False

        if actor.frozen > 0:
            actor.stats.incr(b"status.frozen.ticks")
            actor.stats.incr(b"status.frozen.ticks", actor.group_name)
            actor.frozen -= 1
            return False

        if actor.energy < self.action_cost:
            return False

        actor.update_energy(-self.action_cost, &self.env._rewards[actor_id])
        cdef bint result = self._handle_action(actor_id, actor, arg)

        if result:
            actor.stats.incr(self._stats.action)
            actor.stats.incr(self._stats.action, actor.group_name)

        return result

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False
