from mettagrid.actions.actions cimport MettaActionHandler

cdef class Attack(MettaActionHandler):
    cdef public int damage

    pass
