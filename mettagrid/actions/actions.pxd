
from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.map cimport map

from puffergrid.grid_object cimport TypeId, GridObjectId
from puffergrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport Agent

cdef struct StatNames:
    string action
    string action_energy
    map[TypeId, string] target
    map[TypeId, string] target_energy

cdef class MettaActionHandler(ActionHandler):
    cdef StatNames _stats
    cdef string action_name
    cdef int action_cost

    cdef bint handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg)

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg)
