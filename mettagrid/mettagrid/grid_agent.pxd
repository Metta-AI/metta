# distutils: language=c++

from mettagrid.grid_object cimport GridObject, GridObjectId

cdef cppclass GridAgent(GridObject):
    pass
