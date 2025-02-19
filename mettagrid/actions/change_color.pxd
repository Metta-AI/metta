from mettagrid.action cimport ActionArg
from mettagrid.objects.agent cimport Agent
from mettagrid.actions.actions cimport MettaActionHandler

cdef class ChangeColorAction(MettaActionHandler):
    cdef unsigned char max_arg(self)
    cdef bint _handle_action(self, unsigned int actor_id, Agent* actor, ActionArg arg)
