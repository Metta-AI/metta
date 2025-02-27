from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from mettagrid.objects.converter cimport Converter
from mettagrid.objects.metta_object cimport ObjectConfig

cdef extern from "temple.hpp":
    cdef cppclass Temple(Converter):
        Temple(GridCoord r, GridCoord c, ObjectConfig cfg) except +

        @staticmethod
        vector[string] feature_names()

