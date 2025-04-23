from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector

from mettagrid.grid_object cimport TypeId, GridObjectId
from mettagrid.grid cimport Grid
from mettagrid.objects.agent cimport Agent

ctypedef unsigned int ActionArg

cdef struct StatNames:
    string success
    string first_use
    string failure

    map[TypeId, string] target
    map[TypeId, string] target_first_use
    vector[string] group

cdef class ActionHandler:
    cdef StatNames _stats
    cdef Grid *_grid
    cdef string _action_name
    cdef unsigned char _priority

    cdef void init(self, Grid *grid)

    cdef bint handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg,
        unsigned int current_timestep)

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg)

    cdef unsigned char max_arg(self)

    cpdef string action_name(self)
