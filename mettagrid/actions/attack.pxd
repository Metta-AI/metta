from mettagrid.actions.actions cimport MettaActionHandler

cdef class Attack(MettaActionHandler):
    cdef int damage
