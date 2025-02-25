from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from .metta_object cimport ObjectConfig
from .usable cimport Usable

cdef extern from "generator.hpp":
    cdef cppclass Generator(Usable):
        Generator(GridCoord r, GridCoord c, ObjectConfig cfg) except +

        @staticmethod
        vector[string] feature_names()
