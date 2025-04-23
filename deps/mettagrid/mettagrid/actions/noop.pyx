from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action_handler cimport ActionArg
from mettagrid.action_handler cimport ActionHandler

from mettagrid.objects.agent cimport Agent

cdef class Noop(ActionHandler):
    def __init__(self, cfg: OmegaConf):
        ActionHandler.__init__(self, "noop")

    cdef unsigned char max_arg(self):
        return 0

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        return 1
