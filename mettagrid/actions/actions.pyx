
from libc.stdio cimport printf
from libcpp.string cimport string
from libc.string cimport strcat, strcpy
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
        self._stats.first_use = "action." + action_name + ".first_use"
        
        for t, n in enumerate(ObjectTypeNames):
            self._stats.target[t] = self._stats.action + "." + n
            self._stats.target_energy[t] = self._stats.action_energy + "." + n
            self._stats.target_first_use[t] = self._stats.first_use + "." + n

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
            self.env._stats.agent_add(actor_id, "shield_upkeep", actor.shield_upkeep)
            strcpy(stat_name, actor.group_name.c_str())
            strcat(stat_name, ".shield_upkeep")
            self.env._stats.agent_add(actor_id, stat_name, actor.shield_upkeep)
            self.env._stats.agent_incr(actor_id, "status.shield.ticks")
            strcpy(stat_name, actor.group_name.c_str())
            strcat(stat_name, ".status.shield.ticks")
            self.env._stats.agent_incr(actor_id, stat_name)
            if actor.energy == 0:
                actor.shield = False

        if actor.frozen > 0:
            self.env._stats.agent_incr(actor_id, "status.frozen.ticks")
            strcpy(stat_name, actor.group_name.c_str())
            strcat(stat_name, ".status.frozen.ticks")
            self.env._stats.agent_incr(actor_id, stat_name)
            actor.frozen -= 1
            return False

        if actor.energy < self.action_cost:
            return False

        actor.update_energy(-self.action_cost, &self.env._rewards[actor_id])
        self.env._stats.agent_add(actor_id, self._stats.action_energy.c_str(), self.action_cost)
        strcpy(stat_name, actor.group_name.c_str())
        strcat(stat_name, ".")
        strcat(stat_name, self._stats.action_energy.c_str())
        self.env._stats.agent_add(actor_id, stat_name, self.action_cost)

        cdef bint result = self._handle_action(actor_id, actor, arg)

        if result:
            self.env._stats.agent_incr(actor_id, self._stats.action.c_str())
            self.env._stats.agent_set_once(actor_id, self._stats.first_use.c_str(), self.env._current_timestep)

        return result

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False