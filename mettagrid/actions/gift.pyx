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
from mettagrid.objects.constants cimport GridLayer


cdef class Gift(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "gift")

    cdef unsigned char max_arg(self):
        return 1

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False
