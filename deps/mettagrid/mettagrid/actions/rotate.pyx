from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action_handler cimport ActionArg
from mettagrid.action_handler cimport ActionHandler

from mettagrid.objects.agent cimport Agent

cdef class Rotate(ActionHandler):
    def __init__(self, cfg: OmegaConf):
        ActionHandler.__init__(self, "rotate")

    cdef unsigned char max_arg(self):
        return 3


    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef unsigned short orientation = arg

        actor.orientation = orientation
        return True
