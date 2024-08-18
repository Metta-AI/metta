
from libc.stdio cimport printf

from omegaconf import OmegaConf

from puffergrid.grid_object cimport GridLocation, GridObjectId, GridObject, Orientation
from puffergrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from mettagrid.actions.actions cimport MettaActionHandler

cdef class Move(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "move")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef unsigned short direction = arg
        if direction >= 2:
            return False

        cdef Orientation orientation = <Orientation>((actor.orientation + 2*(direction)) % 4)
        cdef GridLocation old_loc = actor.location
        cdef GridLocation new_loc = self.env._grid.relative_location(old_loc, orientation)
        if not self.env._grid.is_empty(new_loc.r, new_loc.c):
            return False
        return self.env._grid.move_object(actor.id, new_loc)
