from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action cimport ActionArg
from mettagrid.actions.attack cimport Attack
from mettagrid.grid_object cimport GridLocation, Orientation
from mettagrid.objects.agent cimport Agent, InventoryItem
from mettagrid.objects.constants cimport GridLayer

cdef class AttackNearest(Attack):
    def __init__(self, cfg: OmegaConf):
        Attack.__init__(self, cfg, "attack_nearest")

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

        if actor.inventory[InventoryItem.laser] == 0:
            return False

        actor.update_inventory(InventoryItem.laser, -1, &self.env._rewards[actor_id])

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

