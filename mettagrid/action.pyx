from mettagrid.action cimport ActionArg
from mettagrid.grid_env cimport GridEnv
from mettagrid.grid_object cimport GridObjectId

cdef class ActionHandler:
    def __init__(self, string action_name):
        self._action_name = action_name
        self._priority = 0

    cdef void init(self, GridEnv env):
        self.env = env

    cdef bint handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):
        return False

    cdef unsigned char max_arg(self):
        return 0

    cpdef string action_name(self):
        return self._action_name
