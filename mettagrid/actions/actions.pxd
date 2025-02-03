
from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector

from mettagrid.grid_object cimport TypeId, GridObjectId
from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport Agent

cdef struct StatNames:
    string action
    string action_energy
    map[TypeId, string] target
    map[TypeId, string] target_energy
    vector[string] group

cdef class MettaActionHandler(ActionHandler):
    cdef StatNames _stats
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
