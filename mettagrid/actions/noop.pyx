from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.actions.actions cimport MettaActionHandler
from mettagrid.grid_object cimport (
    GridLocation,
    GridObject,
    GridObjectId,
    Orientation
)
from mettagrid.objects.agent cimport Agent

cdef class Noop(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "noop")

    cdef unsigned char max_arg(self):
        return 0

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        return 1
