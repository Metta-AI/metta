
from libc.stdio cimport printf
from libc.string cimport strcat, strcpy

from omegaconf import OmegaConf

from mettagrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from mettagrid.actions.actions cimport MettaActionHandler
from mettagrid.actions.attack cimport Attack

cdef class AttackNearest(Attack):
    def __init__(self, cfg: OmegaConf):
        Attack.__init__(self, cfg, "attack_nearest")
        self._priority = 1

    cdef unsigned char max_arg(self):
        return 0

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef int distance = 0
        cdef int offset = 0
        cdef GridLocation target_loc;
        cdef Agent * agent_target;
        
        # Scan the space to find the nearest agent. Prefer the middle (offset 0) before the edges (offset -1, 1).
        for distance in range(1, 4):
            for offset in range(3):
                if offset == 2:
                    # Sort of a mod 3 operation.
                    offset = -1
                target_loc = self.env._grid.relative_location(
                    actor.location,
                    <Orientation>actor.orientation,
                    distance, offset)

                target_loc.layer = GridLayer.Agent_Layer
                agent_target = <Agent *>self.env._grid.object_at(target_loc)
                if agent_target:
                    return self._handle_target(actor_id, actor, target_loc)

