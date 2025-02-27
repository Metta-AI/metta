from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from .metta_object cimport ObjectConfig
from .converter cimport Converter

cdef extern from "altar.hpp":
    cdef cppclass Altar(Converter):
        Altar(GridCoord r, GridCoord c, ObjectConfig cfg) except +

        @staticmethod
        vector[string] feature_names()
