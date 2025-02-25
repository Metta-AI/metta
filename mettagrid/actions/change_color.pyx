from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.actions.actions cimport MettaActionHandler
from mettagrid.grid_object cimport GridObject
from mettagrid.objects.agent cimport Agent

cdef class ChangeColorAction(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "change_color")

    cdef unsigned char max_arg(self):
        return 3

    cdef bint _handle_action(self, unsigned int actor_id, Agent* actor, ActionArg arg):
        if arg == 0:  # Increment
            if actor.color < 255:
                actor.color += 1
        elif arg == 1:  # Decrement
            if actor.color > 0:
                actor.color -= 1
        elif arg == 2:  # Double
            if actor.color <= 127:
                actor.color *= 2
        elif arg == 3:  # Half
            actor.color = actor.color // 2

        return True
