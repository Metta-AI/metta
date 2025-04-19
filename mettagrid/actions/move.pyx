from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action_handler cimport ActionArg
from mettagrid.action_handler cimport ActionHandler

from mettagrid.grid_object cimport (
    GridLocation,
    GridObject,
    GridObjectId,
    Orientation
)
from mettagrid.objects.agent cimport Agent

cdef class Move(ActionHandler):
    def __init__(self, cfg: OmegaConf):
        ActionHandler.__init__(self, "move")

    cdef unsigned char max_arg(self):
        return 1

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef unsigned short direction = arg

        cdef Orientation orientation = <Orientation>(actor.orientation)
        if direction == 1:
            if orientation == Orientation.Up:
                orientation = Orientation.Down
            elif orientation == Orientation.Down:
                orientation = Orientation.Up
            elif orientation == Orientation.Left:
                orientation = Orientation.Right
            elif orientation == Orientation.Right:
                orientation = Orientation.Left

        cdef GridLocation old_loc = actor.location
        cdef GridLocation new_loc = self._grid.relative_location(old_loc, orientation)
        if not self._grid.is_empty(new_loc.r, new_loc.c):
            return 0
        return self._grid.move_object(actor.id, new_loc)
