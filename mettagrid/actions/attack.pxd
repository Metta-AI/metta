from mettagrid.actions.actions cimport MettaActionHandler

from mettagrid.grid_object cimport GridLocation
from mettagrid.objects.agent cimport Agent

cdef class Attack(MettaActionHandler):
    cdef int damage

    cdef bint _handle_target(
        self,
        unsigned int actor_id,
        Agent * actor,
        GridLocation target_loc)
