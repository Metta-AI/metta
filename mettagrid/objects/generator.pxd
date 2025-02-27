from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from .metta_object cimport ObjectConfig
from .converter cimport Converter

cdef extern from "generator.hpp":
    cdef cppclass Generator(Converter):
        Generator(GridCoord r, GridCoord c, ObjectConfig cfg)

        @staticmethod
        vector[string] feature_names()
