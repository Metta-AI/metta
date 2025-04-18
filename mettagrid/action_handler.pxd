from mettagrid.grid_object cimport GridObjectId
from mettagrid.grid cimport Grid
from libcpp.string cimport string

ctypedef unsigned int ActionArg

cdef class ActionHandler:
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

    cdef unsigned char max_arg(self)

    cpdef string action_name(self)
