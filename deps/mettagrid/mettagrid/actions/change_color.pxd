from mettagrid.action_handler cimport ActionArg
from mettagrid.objects.agent cimport Agent
from mettagrid.action_handler cimport ActionHandler


cdef class ChangeColorAction(ActionHandler):
    cdef unsigned char max_arg(self)
    cdef bint _handle_action(self, unsigned int actor_id, Agent* actor, ActionArg arg)
