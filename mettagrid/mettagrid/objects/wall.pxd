from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from mettagrid.objects.metta_object cimport MettaObject, ObjectConfig

cdef extern from "wall.hpp":
    cdef cppclass Wall(MettaObject):
        Wall(GridCoord r, GridCoord c, ObjectConfig cfg) except +

        @staticmethod
        vector[string] feature_names()
