from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.actions.actions cimport MettaActionHandler
from mettagrid.grid_object cimport GridLocation, Orientation
from mettagrid.objects.agent cimport Agent

cdef class Rotate(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "rotate")

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
