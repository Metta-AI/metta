
from libc.stdio cimport printf

from omegaconf import OmegaConf

from puffergrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from puffergrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from mettagrid.actions.actions cimport MettaActionHandler

cdef class Shield(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "shield")

    cdef unsigned char max_arg(self):
        return 0


    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        if actor.shield:
            actor.shield = False
        elif actor.energy >= actor.shield_upkeep:
            actor.shield = True

        return True


