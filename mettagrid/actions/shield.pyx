
from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action cimport ActionArg
from mettagrid.objects.agent cimport Agent
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
