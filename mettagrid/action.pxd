from mettagrid.grid_object cimport GridObjectId
from mettagrid.grid_env cimport GridEnv
from libcpp.string cimport string

ctypedef unsigned int ActionArg
ctypedef struct Action:
    unsigned int action
    unsigned int arg

cdef class ActionHandler:
    cdef GridEnv env
    cdef string _action_name
    cdef unsigned char _priority

    cdef void init(self, GridEnv env)

    cdef bint handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg)

    cdef unsigned char max_arg(self)

    cpdef string action_name(self)
